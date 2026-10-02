"""可复现性检查：固定 seed 的基准必须给出完全相同的召回率。

跑两次 benchmark（同 seed），断言每个引擎的召回率、门槛召回率、
构建时间在同一量级（时间允许小抖动，质量指标必须**逐位一致**）。

这条检查的意义：召回率是浮点累加的结果，任何一处「顺序不稳定」
（比如遍历一个 dict、用 set 存候选）都会让它出现 1e-15 级别的抖动。
抖动本身无害，但它会让"调参前后对比"失去意义 ——
你不知道 0.9800 → 0.9801 是改进还是噪声。
所以这里强制要求：同 seed 下，质量指标必须逐位一致。
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from app.benchmark import BenchmarkSpec, run_benchmark  # noqa: E402


def main() -> int:
    print("可复现性检查 · 同 seed 两次基准")
    print("-" * 60)

    spec = BenchmarkSpec(count=400, dim=32, n_queries=15, seed=42, k=10)

    first = run_benchmark(spec)
    second = run_benchmark(spec)

    failures = 0
    for a, b in zip(first.engines, second.engines):
        if a.engine != b.engine:
            print(f"  ✗ 引擎顺序不一致: {a.engine} vs {b.engine}")
            failures += 1
            continue
        same_recall = a.recall == b.recall
        same_thr = a.threshold_recall == b.threshold_recall
        time_ok = abs(a.build_seconds - b.build_seconds) < max(
            0.5, a.build_seconds
        )
        status = "ok" if (same_recall and same_thr) else "FAIL"
        print(
            f"  [{status}] {a.label:14s} "
            f"recall {a.recall:.4f}/{b.recall:.4f} · "
            f"thr {a.threshold_recall:.4f}/{b.threshold_recall:.4f} · "
            f"构建 {a.build_seconds:.3f}s/{b.build_seconds:.3f}s"
        )
        if not same_recall:
            print(f"         ✗ 召回率不一致 —— 内部存在顺序不稳定的遍历")
            failures += 1
        if not same_thr:
            print(f"         ✗ 门槛召回率不一致")
            failures += 1
        if not time_ok:
            print(f"         ✗ 构建时间差异过大")
            failures += 1

    print("-" * 60)
    if failures:
        print(f"✗ {failures} 项不一致 —— 质量指标必须逐位复现")
        return 1
    print("✓ 全部逐位一致 —— 同 seed 下结果是确定性的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
