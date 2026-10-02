"""端到端黑盒验收：只通过 HTTP 访问，不 import 任何 app 模块。

和 pytest 的分工很清楚：
pytest 验证**内部正确性**（某个启发式分支对不对），
这个脚本验证**对外承诺是否兑现**（接口给不给、数字自不自洽、
错误码对不对、两次调用是否幂等）。

用 urllib 而不是 requests —— 少一个依赖，
而且这个脚本要在 CI 的一个单独 job 里跑，环境越干净越好。
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8131"

PASSED = 0
FAILED = 0
SKIPPED = 0
_current = ""


# ---------------------------------------------------------------------------
# 断言工具
# ---------------------------------------------------------------------------

def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f"  ← {detail}" if detail else ""))


def close(name: str, a: float, b: float, tol: float, unit: str = "") -> None:
    ok = abs(a - b) <= tol
    check(f"{name} ≈ {b}{unit}", ok, f"实际 {a}{unit}，容差 {tol}{unit}")


def section(title: str) -> None:
    global _current
    _current = title
    print(f"\n[{title}]")


def request(method: str, path: str, body: dict[str, Any] | None = None,
            raw: bool = False) -> tuple[int, Any]:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        DEFAULT_BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = resp.read().decode("utf-8")
            return resp.status, payload if raw else json.loads(payload)
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8")
        try:
            return e.code, json.loads(payload)
        except json.JSONDecodeError:
            return e.code, payload


# ---------------------------------------------------------------------------
# 各段检查
# ---------------------------------------------------------------------------

def section_health() -> None:
    section("1 健康检查")
    code, d = request("GET", "/api/health")
    check("GET /api/health 返回 200", code == 200, f"实际 {code}")
    check("status 为 ok", d.get("status") == "ok", str(d))
    check("version 非空", bool(d.get("version")), str(d))
    check("uptime 为正", d.get("uptime_seconds", 0) >= 0, str(d))


def section_options() -> None:
    section("2 元信息")
    code, d = request("GET", "/api/options")
    check("GET /api/options 返回 200", code == 200, f"实际 {code}")
    for key in ("metrics", "kinds", "engines", "bounds"):
        check(f"包含 {key}", key in d, str(d))
    check("metrics 含 l2/cosine/inner",
          set(["l2", "cosine", "inner"]).issubset(set(d.get("metrics", []))), str(d))
    b = d.get("bounds", {})
    check("count 上界为 20000", b.get("count", [0, 0])[1] == 20000, str(b))
    check("dim 上界为 512", b.get("dim", [0, 0])[1] == 512, str(b))


def section_dataset() -> str:
    section("3 数据集")
    code, d = request("POST", "/api/dataset", {
        "count": 600, "dim": 32, "metric": "l2", "kind": "clustered",
        "seed": 42, "n_clusters": 24, "overlap": 0.15,
    })
    check("POST /api/dataset 返回 200", code == 200, f"{code}: {d}")
    ds_id = d.get("id", "")
    check("返回 12 位 id", len(ds_id) == 12, ds_id)
    ds = d.get("dataset", {})
    check("count 回显 600", ds.get("count") == 600, str(ds))
    check("dim 回显 32", ds.get("dim") == 32, str(ds))
    check("seed 回显 42", ds.get("seed") == 42, str(ds))
    check("revision = count", d.get("store_revision") == 600, str(d))
    # 32 维 float32 = 128 字节/向量
    check("向量内存 = 600×128", d.get("vector_bytes") == 600 * 128,
          str(d.get("vector_bytes")))
    check("初始没有已建索引", d.get("built_indexes") == [], str(d))

    # 幂等：同参数两次生成应当是两组不同的 id（各自独立）
    code2, d2 = request("POST", "/api/dataset", {
        "count": 600, "dim": 32, "metric": "l2", "kind": "clustered",
        "seed": 42, "n_clusters": 24, "overlap": 0.15,
    })
    check("同参数再生成得到不同 id", d2.get("id") != ds_id, str(d2))

    code3, list_d = request("GET", "/api/dataset")
    check("GET /api/dataset 返回列表", code3 == 200 and len(list_d.get("items", [])) >= 2,
          str(list_d)[:200])

    code4, got = request("GET", f"/api/dataset/{ds_id}")
    check("按 id 取回成功", code4 == 200 and got.get("id") == ds_id, str(got)[:200])
    return ds_id


def section_invalid_dataset() -> None:
    section("4 参数校验")
    code, d = request("POST", "/api/dataset", {"count": 5, "dim": 32})
    check("count 过小 → 422", code == 422, f"{code}: {d}")

    code, d = request("POST", "/api/dataset", {"count": 100, "dim": 9999})
    check("dim 过大 → 422", code == 422, f"{code}: {d}")

    code, d = request("POST", "/api/dataset", {"count": 100, "dim": 32, "kind": "spiral"})
    check("未知 kind → 422", code == 422, f"{code}: {d}")

    code, d = request("POST", "/api/dataset", {"count": 100, "dim": 32, "metric": "manhattan"})
    check("未知 metric → 422", code == 422, f"{code}: {d}")

    code, d = request("GET", "/api/dataset/does-not-exist")
    check("不存在的数据集 → 404（不是 500）", code == 404, f"{code}: {d}")


def section_hnsw(ds_id: str) -> None:
    section("5 HNSW 索引")
    code, d = request("POST", f"/api/index/{ds_id}", {
        "engine": "hnsw",
        "params": {"m": 16, "ef_construction": 100, "ef_search": 64},
    })
    check("构建返回 200", code == 200, f"{code}: {d}")
    check("构建耗时为正", d.get("build_wall_seconds", 0) > 0, str(d))
    s = d.get("snapshot", {})

    check("节点数 = 600", s.get("nodes") == 600, str(s.get("nodes")))
    check("层数 ≥ 1", s.get("levels", 0) >= 1, str(s.get("levels")))

    per_level = s.get("nodes_per_level", [])
    check("第 0 层是全部节点", per_level and per_level[0] == 600, str(per_level))
    check("层数递减", all(per_level[i] > per_level[i + 1]
                        for i in range(len(per_level) - 1)) or len(per_level) == 1,
          str(per_level))

    cfg = s.get("config", {})
    mem = s.get("memory", {})
    check("内存各项之和 = total",
          mem.get("vectors", 0) + mem.get("links", 0) + mem.get("level_table", 0)
          == mem.get("total", -1), str(mem))
    # 边开销与 ``8 × 平均槽位数`` 必须自洽 —— 这是内存账的基本恒等式
    check("links 字节 = link_slots × 8",
          mem.get("links") == mem.get("link_slots") * 8, str(mem))
    # 平均边槽位通常**远低于** M0=32 的上限：多样性启发式会主动丢掉
    # 大量"和已有邻居指向同一方向"的候选边。这个差距正是启发式起作用的证据，
    # 而不是"图没建满"。断言的是结构性范围，不假设它接近上限。
    avg_slots = mem.get("link_slots", 0) / 600
    check("平均边槽位落在 (1, M0] 区间内",
          1 < avg_slots <= cfg.get("m0", 0),
          f"{avg_slots:.2f} vs m0={cfg.get('m0')}")
    check("平均边槽位确实低于 M0（启发式在剪枝）",
          avg_slots < cfg.get("m0", 1),
          f"{avg_slots:.2f} vs m0={cfg.get('m0')}")
    # "边和向量谁更占"取决于 M0 与 d/2 的比较，不是无条件成立。
    # 这里只断言它们是**同一量级** —— 这才是真正要传达的事实。
    check("图结构与向量本体同量级（各占 ≥20%）",
          mem.get("links", 0) >= mem.get("total", 1) * 0.2
          and mem.get("vectors", 0) >= mem.get("total", 1) * 0.2,
          str(mem))

    # 这一条最容易写错：分子必须是第 0 层的边数，不是所有层之和
    check("layer0_edges ≤ edges", s.get("layer0_edges", 10**9) <= s.get("edges", 0),
          f"{s.get('layer0_edges')} vs {s.get('edges')}")
    expected_deg = round(s.get("layer0_edges", 0) / max(s.get("nodes"), 1), 3)
    check("avg_degree_layer0 与 layer0_edges 自洽",
          abs(s.get("avg_degree_layer0", -1) - expected_deg) < 0.01,
          f"{s.get('avg_degree_layer0')} vs {expected_deg}")

    check("m0 = 2m", cfg.get("m0") == cfg.get("m") * 2, str(cfg))
    # 快照里的值是 round(...,4)，容差要匹配这个精度而不是浮点的精度
    close("level_multiplier = 1/ln(m)", cfg.get("level_multiplier", 0),
          1.0 / 2.772588722239781, 1e-4)

    check("构建期距离计算已单独记账",
          s.get("construction_computations", 0) > 0,
          str(s.get("construction_computations")))
    check("未查询时 avg_distance_per_query = 0",
          s.get("avg_distance_per_query", -1) == 0,
          str(s.get("avg_distance_per_query")))

    # 幂等：重复构建同一个引擎，结果应当一致（固定 seed）
    code2, d2 = request("POST", f"/api/index/{ds_id}", {
        "engine": "hnsw",
        "params": {"m": 16, "ef_construction": 100, "ef_search": 64},
    })
    check("重建后边数一致（seed 可复现）",
          d2.get("snapshot", {}).get("edges") == s.get("edges"),
          f"{d2.get('snapshot', {}).get('edges')} vs {s.get('edges')}")

    code3, snap = request("GET", f"/api/index/{ds_id}/hnsw")
    check("GET 快照返回 200", code3 == 200, f"{code3}")
    check("GET 快照与构建结果一致",
          snap.get("snapshot", {}).get("nodes") == s.get("nodes"), str(snap)[:200])


def section_ivfpq(ds_id: str) -> None:
    section("6 IVF-PQ 索引")
    code, d = request("POST", f"/api/index/{ds_id}", {
        "engine": "ivfpq",
        "params": {"nlist": 16, "m": 8, "nbits": 4, "nprobe": 4},
    })
    check("构建返回 200", code == 200, f"{code}: {d}")
    s = d.get("snapshot", {})
    check("trained = 600", s.get("trained") == 600, str(s.get("trained")))
    check("nlist = 16", s.get("nlist") == 16, str(s.get("nlist")))

    ls = s.get("list_sizes", {})
    check("倒排表尺寸有统计", all(k in ls for k in ("min", "max", "empty", "avg")), str(ls))
    check("各表点数之和 = 600",
          ls.get("avg", 0) * 16 == 600, str(ls))

    mem = s.get("memory", {})
    check("索引不保存原始向量（无 vectors 字段）", "vectors" not in mem, str(mem))
    check("codes_only_compression = 4d/m = 16",
          abs(mem.get("codes_only_compression", 0) - 16.0) < 0.01, str(mem))
    check("codes_only ≥ index_total（固定开销的存在）",
          mem.get("codes_only_compression", 0) >= mem.get("index_total_compression", 0),
          str(mem))
    check("内存各项之和 = total",
          mem.get("compressed_codes", 0) + mem.get("codebooks", 0)
          + mem.get("coarse_centroids", 0) + mem.get("inverted_lists", 0)
          == mem.get("total", -1), str(mem))

    cfg = s.get("config", {})
    check("codebook_size = 2^nbits", cfg.get("codebook_size") == 16, str(cfg))
    check("sub_dim = dim/m", cfg.get("sub_dim") == 32 // 8, str(cfg))

    # m 不能整除 dim 时必须报错
    code2, d2 = request("POST", f"/api/index/{ds_id}", {
        "engine": "ivfpq", "params": {"nlist": 8, "m": 5, "nbits": 4, "nprobe": 2},
    })
    check("m 不整除 dim → 422", code2 == 422, f"{code2}: {d2}")

    # nbits > 8 必须报错
    code3, d3 = request("POST", f"/api/index/{ds_id}", {
        "engine": "ivfpq", "params": {"nlist": 8, "m": 8, "nbits": 9, "nprobe": 2},
    })
    check("nbits > 8 → 422", code3 == 422, f"{code3}: {d3}")

    code4, d4 = request("POST", f"/api/index/{ds_id}", {"engine": "nope", "params": {}})
    check("未知引擎 → 422", code4 == 422, f"{code4}: {d4}")


def section_search(ds_id: str) -> None:
    section("7 单次检索")
    code, d = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "hnsw", "k": 10, "probe_index": 0,
    })
    check("检索返回 200", code == 200, f"{code}: {d}")
    for key in ("ann", "exact", "recall", "threshold_recall", "latency"):
        check(f"响应含 {key}", key in d, str(list(d)))
    check("ANN 结果恰好 k 个", len(d.get("ann", {}).get("keys", [])) == 10, str(d))
    check("精确结果恰好 k 个", len(d.get("exact", {}).get("keys", [])) == 10, str(d))
    check("HNSW 召回率 ≥ 0.95", d.get("recall", 0) >= 0.95, str(d.get("recall")))

    dist = d.get("exact", {}).get("distances", [])
    check("精确距离升序", dist == sorted(dist), str(dist))

    ann_dist = d.get("ann", {}).get("distances", [])
    check("ANN 距离升序", ann_dist == sorted(ann_dist), str(ann_dist))

    lat = d.get("latency", {})
    check("speedup 与两个延迟自洽",
          abs(lat.get("speedup", 0) - lat.get("exact_ms", 0) / max(lat.get("ann_ms", 1), 1e-9)) < 0.15,
          str(lat))

    check("probe_label 非空", bool(d.get("probe_label")), str(d))

    # 覆盖 ef_search
    code2, d2 = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "hnsw", "k": 10, "probe_index": 0, "ef_search": 256,
    })
    check("ef_search 覆盖被回显",
          d2.get("overrides", {}).get("ef_search") == 256, str(d2.get("overrides")))
    check("大 ef 召回率不低于小 ef",
          d2.get("recall", 0) >= d.get("recall", 0),
          f"{d2.get('recall')} vs {d.get('recall')}")

    # IVF-PQ：两种召回率
    code3, d3 = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "ivfpq", "k": 10, "probe_index": 0, "nprobe": 16,
    })
    check("IVF-PQ 检索返回 200", code3 == 200, f"{code3}: {d3}")
    check("IVF-PQ 门槛召回 ≥ 标准召回",
          d3.get("threshold_recall", 0) >= d3.get("recall", 0) - 1e-9,
          f"std={d3.get('recall')} thr={d3.get('threshold_recall')}")
    check("两种召回率都在 [0,1]",
          0 <= d3.get("recall", -1) <= 1 and 0 <= d3.get("threshold_recall", -1) <= 1, str(d3))

    # 幂等
    code4, d4 = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "hnsw", "k": 10, "probe_index": 0,
    })
    check("重复检索结果完全一致",
          d4.get("ann", {}).get("keys") == d.get("ann", {}).get("keys"),
          f"{d4.get('ann', {}).get('keys')} vs {d.get('ann', {}).get('keys')}")

    code5, d5 = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "hnsw", "k": 10, "probe_index": 10**6,
    })
    check("越界 probe_index 被取模而非 500",
          code5 == 200 and d5.get("ann"), f"{code5}: {d5}")

    code6, d6 = request("POST", "/api/search", {
        "dataset_id": "nonexistent-id", "engine": "hnsw", "k": 10,
    })
    check("未知数据集 → 404", code6 == 404, f"{code6}: {d6}")


def section_inner_product(ds_id: str) -> None:
    section("8 内积与 IVF-PQ 的兼容性")
    code, d = request("POST", "/api/dataset", {
        "count": 200, "dim": 16, "metric": "inner", "kind": "clustered", "seed": 7,
    })
    check("内积数据集可生成", code == 200, f"{code}: {d}")
    ip_id = d.get("id", "")

    code2, d2 = request("POST", f"/api/index/{ip_id}", {
        "engine": "ivfpq", "params": {"nlist": 8, "m": 4, "nbits": 4, "nprobe": 2},
    })
    check("IVF-PQ 在内积度量下被拒绝（422 不 500）", code2 == 422, f"{code2}: {d2}")
    check("错误信息指明了替代方案", "cosine" in str(d2) or "IVF-PQ" in str(d2), str(d2))

    code3, d3 = request("POST", f"/api/index/{ip_id}", {
        "engine": "hnsw", "params": {"m": 8, "ef_construction": 40, "ef_search": 32},
    })
    check("HNSW 支持内积度量", code3 == 200, f"{code3}: {d3}")

    request("DELETE", f"/api/dataset/{ip_id}")


def section_benchmark() -> None:
    section("9 基准评测")
    code, d = request("POST", "/api/benchmark", {
        "count": 600, "dim": 32, "k": 10, "n_queries": 20, "seed": 42, "sweep": True,
    })
    check("基准返回 200", code == 200, f"{code}: {str(d)[:300]}")
    check("含 spec 与 result", "spec" in d and "result" in d, str(list(d)))
    res = d.get("result", {})
    engines = {e["engine"]: e for e in res.get("engines", [])}

    check("三个引擎都出现", set(engines) == {"brute", "hnsw", "ivfpq"}, str(list(engines)))
    check("没有引擎报错", all(not e.get("error") for e in engines.values()),
          str({k: v.get("error") for k, v in engines.items() if v.get("error")}))

    brute = engines.get("brute", {})
    check("暴力检索召回率为 1.0", abs(brute.get("recall", 0) - 1.0) < 1e-9, str(brute))
    check("暴力门槛召回率为 1.0", abs(brute.get("threshold_recall", 0) - 1.0) < 1e-9, str(brute))
    check("暴力每次查询扫描全部向量",
          brute.get("scanned_per_query") == 600, str(brute.get("scanned_per_query")))

    hnsw = engines.get("hnsw", {})
    check("HNSW 召回率 ≥ 0.9", hnsw.get("recall", 0) >= 0.9, str(hnsw.get("recall")))
    # 这一条曾经因为没有扣除构建期的距离计算而失败：
    # 那时 HNSW 的"每次查询距离计算"比暴力检索还高得多，结论自相矛盾
    check("HNSW 每次查询距离计算少于暴力（扣除构建期后）",
          hnsw.get("scanned_per_query", 10**9) < brute.get("scanned_per_query", 0),
          f"hnsw={hnsw.get('scanned_per_query')} brute={brute.get('scanned_per_query')}")

    ivf = engines.get("ivfpq", {})
    check("IVF-PQ 门槛召回 ≥ 标准召回",
          ivf.get("threshold_recall", 0) >= ivf.get("recall", 0) - 1e-9,
          f"std={ivf.get('recall')} thr={ivf.get('threshold_recall')}")
    check("IVF-PQ 内存远小于向量本体",
          ivf.get("memory", {}).get("total", 10**9) < brute.get("memory", {}).get("total", 0),
          f"ivf={ivf.get('memory')} brute={brute.get('memory')}")

    for name, e in engines.items():
        lat = e.get("latency", {})
        check(f"{name} 延迟分位数单调（p50≤p95≤p99）",
              lat.get("p50_ms", 0) <= lat.get("p95_ms", 0) <= lat.get("p99_ms", 0) + 1e-9,
              str(lat))
        check(f"{name} 有 20 个延迟样本", lat.get("samples") == 20, str(lat))

    sweep_h = i = hnsw.get("sweep", [])
    check("HNSW 有 ef 扫描点", len(sweep_h) >= 4, str(len(sweep_h)))
    if sweep_h:
        check("HNSW 扫描点召回率不降",
              all(sweep_h[j]["recall"] <= sweep_h[j + 1]["recall"] + 1e-9
                  for j in range(len(sweep_h) - 1)),
              str([round(s["recall"], 3) for s in sweep_h]))
        check("HNSW 扫描点延迟随 ef 上升",
              sweep_h[-1]["mean_ms"] > sweep_h[0]["mean_ms"],
              str([round(s["mean_ms"], 3) for s in sweep_h]))

    sweep_i = ivf.get("sweep", [])
    check("IVF-PQ 有 nprobe 扫描点", len(sweep_i) >= 3, str(len(sweep_i)))
    if sweep_i:
        check("IVF-PQ 扫描点召回率不降",
              all(sweep_i[j]["recall"] <= sweep_i[j + 1]["recall"] + 1e-9
                  for j in range(len(sweep_i) - 1)),
              str([round(s["recall"], 3) for s in sweep_i]))

    # GET 版本
    code2, d2 = request("GET", "/api/benchmark?count=200&dim=16&n_queries=5&sweep=false")
    check("GET /api/benchmark 可用", code2 == 200, f"{code2}")
    check("sweep=false 时没有扫描点",
          all(not e.get("sweep") for e in d2.get("result", {}).get("engines", [])),
          str(d2)[:200])


def section_errors(ds_id: str) -> None:
    section("10 错误处理")
    code, d = request("GET", f"/api/index/{ds_id}/never-built")
    check("未构建的引擎 → 404", code == 404, f"{code}: {d}")

    code, d = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "hnsw", "k": 10, "probe_index": -1,
    })
    check("负 probe_index → 422", code == 422, f"{code}: {d}")

    code, d = request("POST", "/api/search", {
        "dataset_id": ds_id, "engine": "hnsw", "k": 0,
    })
    check("k=0 → 422", code == 422, f"{code}: {d}")

    code, page = request("GET", "/", raw=True)
    check("根路径返回前端页面",
          code == 200 and "<!DOCTYPE HTML>".upper() in str(page).upper(),
          str(page)[:120])


def main() -> int:
    global DEFAULT_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()
    DEFAULT_BASE = args.base.rstrip("/")

    print("=" * 70)
    print(f"vector-index · 端到端验收  {DEFAULT_BASE}")
    print("=" * 70)

    try:
        section_health()
        section_options()
        ds_id = section_dataset()
        section_invalid_dataset()
        section_hnsw(ds_id)
        section_ivfpq(ds_id)
        section_search(ds_id)
        section_inner_product(ds_id)
        section_benchmark()
        section_errors(ds_id)
        request("DELETE", f"/api/dataset/{ds_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n段落 [{_current}] 抛出异常：{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        global FAILED
        FAILED += 1

    print("\n" + "=" * 70)
    print(f"通过 {PASSED} · 失败 {FAILED} · 跳过 {SKIPPED}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
