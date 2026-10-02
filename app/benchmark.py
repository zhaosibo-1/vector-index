"""基准评测：把"哪一种索引更好"变成可复核的数字。

设计上刻意遵守三件事：

**1. 召回率给两个数字。**
标准召回率（集合交集）和门槛召回率（距离是否够近）在并列多的时候
会给出不同的答案。一个宣称"我用 XX 索引召回率 0.99"的报告，
如果没说清是哪种口径、数据里有多少并列，这个数字不可复核。
这里两个都给，并且让差异可见。

**2. 延迟给分位数，不给平均值。**
p95 和 p50 之间的差距有多大，决定了这个索引能不能上线。
平均值会被大量的容易查询拉低，掩盖最坏情况。

**3. 扫描 sweep 而不是单点。**
效率最高的结论是"把 ef 从 32 调到 64，召回率从 0.92 到 0.98，
代价是延迟翻倍"这种曲线关系。给一个 ef=64 的孤立数字，
读者无法据此做任何决策。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from .bruteforce import exact_top_k, recall_at_k, threshold_recall_at_k
from .dataset import Dataset, generate, generate_queries
from .hnsw import HNSWConfig, HNSWIndex
from .ivfpq import IVFPQConfig, IVFPQIndex
from .metrics import Metric, VectorStore, make_metric

#: HNSW 的 ef_search 扫描点。从 16 起每隔一档翻倍到 256 ——
#: 覆盖"极致低延迟"到"精度优先"的完整区间。
HNSW_EF_SWEEP: tuple[int, ...] = (16, 32, 64, 128, 256)

#: IVF-PQ 的 nprobe 扫描点。取 nlist 的若干比例，因为 nprobe 的
#: 合理取值范围取决于 nlist —— 给绝对值会在小 nlist 上溢出、
#: 在大 nlist 上等于没扫。
IVFPQ_NPROBE_RATIOS: tuple[float, ...] = (0.06, 0.12, 0.25, 0.5, 1.0)


@dataclass
class BenchmarkSpec:
    """一次基准评测的参数。"""

    count: int = 1000
    dim: int = 32
    kind: str = "clustered"
    metric: str = "l2"
    seed: int = 42
    n_clusters: int = 24
    overlap: float = 0.15
    k: int = 10
    n_queries: int = 30
    #: 参与评测的索引类型
    engines: tuple[str, ...] = ("brute", "hnsw", "ivfpq")
    #: HNSW 参数
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 64
    #: IVF-PQ 参数
    ivfpq_nlist: int = 16
    ivfpq_m: int = 8
    ivfpq_nbits: int = 4
    ivfpq_nprobe: int = 4
    #: 是否做 ef / nprobe 的单点扫描
    sweep: bool = True

    def describe(self) -> dict[str, object]:
        return dict(self.__dict__) | {"engines": list(self.engines)}


@dataclass
class LatencyStats:
    """延迟统计，单位毫秒。"""

    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    #: 每秒能处理多少查询，由平均延迟换算
    qps: float = 0.0
    samples: int = 0


@dataclass
class EngineResult:
    """单个引擎的评测结果。"""

    engine: str
    label: str
    build_seconds: float = 0.0
    #: 标准召回率（集合交集 / k）
    recall: float = 0.0
    #: 门槛召回率（距离够近的比例）
    threshold_recall: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    distance_computations: int = 0
    #: 平均每次查询扫过多少候选（brute 是全部，ivfpq 是 nprobe 个表）
    scanned_per_query: float = 0.0
    memory: dict[str, int] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    #: 扫描曲线：``[(旋钮值, 召回率, 平均延迟ms)]``
    sweep: list[dict[str, float]] = field(default_factory=list)
    error: str = ""

    def to_wire(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "label": self.label,
            "build_seconds": round(self.build_seconds, 4),
            "recall": round(self.recall, 4),
            "threshold_recall": round(self.threshold_recall, 4),
            "latency": {
                "mean_ms": round(self.latency.mean_ms, 4),
                "p50_ms": round(self.latency.p50_ms, 4),
                "p95_ms": round(self.latency.p95_ms, 4),
                "p99_ms": round(self.latency.p99_ms, 4),
                "max_ms": round(self.latency.max_ms, 4),
                "qps": round(self.latency.qps, 1),
                "samples": self.latency.samples,
            },
            "distance_computations": self.distance_computations,
            "scanned_per_query": round(self.scanned_per_query, 1),
            "memory": self.memory,
            "params": self.params,
            "sweep": self.sweep,
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    dataset: dict[str, object] = field(default_factory=dict)
    engines: list[EngineResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_wire(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "engines": [e.to_wire() for e in self.engines],
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "warnings": self.warnings,
        }


def _latency(samples_ms: list[float]) -> LatencyStats:
    """从样本算分位数。

    用**线性插值**而不是"取下整处的值"：只有 20 个样本时，
    后者会让 p95 退化成最大值，而 p99 完全等于 max ——
    那时候 p95 和 p99 是同一个数，读者会误以为尾部很平。
    """
    if not samples_ms:
        return LatencyStats()
    ordered = sorted(samples_ms)
    n = len(ordered)

    def percentile(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = p * (n - 1)
        low = int(pos)
        high = min(low + 1, n - 1)
        frac = pos - low
        return ordered[low] * (1 - frac) + ordered[high] * frac

    mean = statistics.fmean(ordered)
    return LatencyStats(
        mean_ms=mean,
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        max_ms=ordered[-1],
        qps=1000.0 / mean if mean > 0 else 0.0,
        samples=n,
    )


def _measure(index, queries, prepared, store, metric, k, **search_kw):
    """跑一遍查询，返回（标准召回, 门槛召回, 延迟样本ms, 扫描数）。"""
    recall_sum = 0.0
    threshold_sum = 0.0
    samples: list[float] = []
    scanned = 0.0

    for query in prepared:
        started = time.perf_counter()
        hits = index.search(query[0], query[1], k=k, **search_kw)
        elapsed = (time.perf_counter() - started) * 1000.0
        samples.append(elapsed)

        from .bruteforce import exact_top_k as _exact

        exact = _exact(store, query, metric, k=k)
        recall_sum += recall_at_k([key for _, key in hits], exact)
        threshold_sum += threshold_recall_at_k(hits, exact)

    n = len(prepared)
    return recall_sum / n, threshold_sum / n, samples, scanned


def _build_store(dataset: Dataset, metric: Metric) -> VectorStore:
    store = VectorStore(dataset.dim, metric)
    for index, vector in enumerate(dataset.vectors):
        store.add(vector, label=dataset.labels[index] if dataset.labels else "")
    return store


def run_benchmark(spec: BenchmarkSpec) -> BenchmarkResult:
    """按 spec 跑一次完整评测。"""
    started = time.perf_counter()
    metric = make_metric(spec.metric)

    dataset = generate(
        count=spec.count,
        dim=spec.dim,
        kind=spec.kind,
        metric=spec.metric,
        seed=spec.seed,
        n_clusters=spec.n_clusters,
        overlap=spec.overlap,
    )
    store = _build_store(dataset, metric)

    raw_queries = generate_queries(dataset, n_queries=spec.n_queries, seed=spec.seed + 1)
    prepared = [store.prepare_query(q) for q in raw_queries]

    result = BenchmarkResult()
    result.dataset = dataset.describe()
    result.dataset["queries"] = spec.n_queries
    result.dataset["k"] = spec.k

    if spec.count <= 2 * spec.k:
        result.warnings.append(
            f"数据集只有 {spec.count} 个点，而 k={spec.k}："
            "top-k 的结果会被库的大小压住，召回率失去参考价值。"
        )

    for engine in spec.engines:
        entry = EngineResult(engine=engine, label=_label(engine))
        try:
            if engine == "brute":
                _run_brute(entry, store, metric, prepared, spec)
            elif engine == "hnsw":
                _run_hnsw(entry, store, metric, prepared, spec)
            elif engine == "ivfpq":
                _run_ivfpq(entry, store, metric, prepared, spec)
            else:
                entry.error = f"未知的引擎 {engine!r}"
        except Exception as exc:  # noqa: BLE001
            # 评测整体不能因为一个引擎失败就全盘失败：
            # 用户想看的是"哪个能用"，而错误信息本身也是结论的一部分。
            entry.error = f"{type(exc).__name__}: {exc}"
        if entry.error:
            result.warnings.append(f"{engine}: {entry.error}")
        result.engines.append(entry)

    result.elapsed_seconds = time.perf_counter() - started
    return result


def _label(engine: str) -> str:
    return {
        "brute": "暴力检索（精确）",
        "hnsw": "HNSW（图）",
        "ivfpq": "IVF-PQ（量化）",
    }.get(engine, engine)


def _run_brute(entry, store, metric, prepared, spec):
    """暴力检索：召回率恒为 1，它是**基准**而不是被评测对象。

    它的价值在于给出"正确但不实用"的一端：
    延迟和后面的 ANN 对比，才有了"快了多少倍"的参照。
    """
    recalls: list[float] = []
    samples: list[float] = []

    for query in prepared:
        started = time.perf_counter()
        exact = exact_top_k(store, query, metric, k=spec.k)
        elapsed = (time.perf_counter() - started) * 1000.0
        samples.append(elapsed)
        recalls.append(recall_at_k(exact.keys, exact))

    entry.build_seconds = 0.0
    entry.recall = statistics.fmean(recalls)
    entry.threshold_recall = 1.0
    entry.latency = _latency(samples)
    entry.scanned_per_query = float(len(store))
    entry.memory = {"vectors": store.memory_bytes(), "total": store.memory_bytes()}
    entry.params = {"k": spec.k}
    entry.distance_computations = len(prepared) * len(store)


def _run_hnsw(entry, store, metric, prepared, spec):
    config = HNSWConfig(
        m=spec.hnsw_m,
        ef_construction=spec.hnsw_ef_construction,
        ef_search=spec.hnsw_ef_search,
        seed=spec.seed,
    )
    index = HNSWIndex(store, metric, config)

    build_started = time.perf_counter()
    index.build()
    entry.build_seconds = time.perf_counter() - build_started

    recall, threshold, samples, _ = _measure(
        index, None, prepared, store, metric, spec.k
    )
    entry.recall = recall
    entry.threshold_recall = threshold
    entry.latency = _latency(samples)
    entry.memory = index.memory_bytes()
    entry.distance_computations = index.stats.distance_computations
    # "平均每次查询算了多少次距离" —— HNSW 没有独立的扫描表，
    # 这个数字就是它的实际工作量。它比延迟更能说明
    # 一次搜索是走了多少冤枉路。
    #
    # 必须扣掉构建期：``distance_computations`` 是构建与查询的累加值，
    # 直接用总量会得到比暴力检索还高的荒谬数字。
    entry.scanned_per_query = (
        index.stats.query_computations / max(index.stats.queries, 1)
    )
    entry.params = {
        "m": config.m,
        "ef_construction": config.ef_construction,
        "ef_search": config.ef_search,
        "levels": stats_levels(index),
        "avg_degree_layer0": avg_degree(index),
    }

    if spec.sweep:
        for ef in HNSW_EF_SWEEP:
            r, _t, sweep_samples, _s = _measure(
                index, None, prepared, store, metric, spec.k, ef_search=ef
            )
            entry.sweep.append(
                {
                    "knob": "ef_search",
                    "value": ef,
                    "recall": round(r, 4),
                    "mean_ms": round(statistics.fmean(sweep_samples), 4),
                }
            )


def _run_ivfpq(entry, store, metric, prepared, spec):
    config = IVFPQConfig(
        nlist=spec.ivfpq_nlist,
        m=spec.ivfpq_m,
        nbits=spec.ivfpq_nbits,
        nprobe=spec.ivfpq_nprobe,
        seed=spec.seed,
    )
    index = IVFPQIndex(store, metric, config)

    build_started = time.perf_counter()
    index.build()
    entry.build_seconds = time.perf_counter() - build_started

    recall, threshold, samples, _ = _measure(
        index, None, prepared, store, metric, spec.k
    )
    entry.recall = recall
    entry.threshold_recall = threshold
    entry.latency = _latency(samples)
    entry.memory = index.memory_bytes()
    entry.scanned_per_query = (
        index.stats.scanned_total / index.query_invocations
    )
    entry.params = {
        "nlist": index.config.nlist,
        "actual_lists": len(index._coarse),
        "m": config.m,
        "nbits": config.nbits,
        "nprobe": index.config.nprobe,
        "compression_ratio": index.memory_bytes()["compression_ratio"],
    }

    if spec.sweep:
        actual = len(index._coarse)
        seen: set[int] = set()
        for ratio in IVFPQ_NPROBE_RATIOS:
            nprobe = max(1, int(round(actual * ratio)))
            if nprobe in seen:
                continue
            seen.add(nprobe)
            r, _t, sweep_samples, _s = _measure(
                index, None, prepared, store, metric, spec.k, nprobe=nprobe
            )
            entry.sweep.append(
                {
                    "knob": "nprobe",
                    "value": nprobe,
                    "recall": round(r, 4),
                    "mean_ms": round(statistics.fmean(sweep_samples), 4),
                }
            )


def stats_levels(index) -> int:
    return index.stats.levels


def avg_degree(index) -> float:
    return round(
        index.stats.layer0_edges / max(index.stats.nodes, 1), 2
    )
