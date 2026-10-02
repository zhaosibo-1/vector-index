"""暴力精确 kNN —— 召回率的唯一真值来源。

这个模块不追求快（它就是"慢"的基准），追求的是**结果无歧义**。
因为整个项目的核心指标"召回率"是拿它的输出当分母算出来的，
真值本身有歧义的话，后面所有结论都不成立。

==========================================================================
 并列最近邻：召回率的分母到底是什么
==========================================================================
这是评测里一个真实存在、但大多数实现都含糊过去的问题。

``duplicates`` 数据集里有大量"近乎重复"的点。当第 k 名和第 k+1 名距离
完全一样时，"精确 top-k" 是**没有唯一定义**的 —— 取哪一组取决于
排序实现里 tie-break 的细节（谁先被遍历到）。

于是：
* ANN 返回了 {A, B, C, D}，精确实现因为遍历顺序返回了 {A, B, C, E}，
  交集是 3，召回率算成 0.75 —— **但这个索引其实完全正确**，
  D 和 E 与查询点的距离一模一样。
* 反过来，如果 ANN 恰好和某个特定 tie-break 的结果一致，
  召回率会显得比真实能力好。

两种应对，本项目都提供：

1. ``recall_at_k``  —— 集合交集 / k。这是**标准定义**，几乎所有论文和
   产品都这么算。作为默认指标，因为它可比。
2. ``threshold_recall_at_k`` —— 数 ANN 结果里有多少个的**距离不超过
   第 k 名的距离**。这是"在某种合法的 tie-break 下算对"的口径，
   在有大量并列时更能反映真实能力。

两个数一起看：如果它们差距很大，说明这个数据集里并列很多，
**这时候拿标准召回率去比较两个索引是不可靠的**。基准结果里会同时给出来。
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import Metric, VectorStore, top_k


@dataclass(frozen=True)
class ExactResult:
    """一次精确 kNN 的结果。"""

    #: 按距离升序的 ``(距离, key)``
    hits: list[tuple[float, int]]
    metric: str
    k: int

    @property
    def keys(self) -> list[int]:
        return [key for _, key in self.hits]

    @property
    def distances(self) -> list[float]:
        return [distance for distance, _ in self.hits]

    @property
    def threshold(self) -> float:
        """第 k 名的距离，即"进得了 top-k 的门槛"。

        不足 k 个结果（库里的点比 k 还少）时返回 ``inf`` ——
        表示"所有的点都算命中"。返回 0 是错的：那会让所有距离大于 0 的
        ANN 结果都被判为未命中，召回率恒等于 0。
        """
        if not self.hits:
            return float("inf")
        if len(self.hits) < self.k:
            return float("inf")
        return self.hits[-1][0]

    def to_wire(self, limit: int = 10) -> list[dict[str, object]]:
        return [
            {"distance": round(d, 6), "key": key}
            for d, key in self.hits[:limit]
        ]


def exact_top_k(
    store: VectorStore,
    query: object,
    metric: Metric,
    *,
    k: int = 10,
) -> ExactResult:
    """对库里所有向量逐个算距离，取最小的 k 个。

    ``query`` 必须是**已经过 ``store.prepare_query`` 处理**的向量
    （归一化过的），理由见 metrics.VectorStore.prepare_query：
    两侧预处理不一致会让 cosine 算出大于 1 的值。

    遍历顺序按 key 升序（``store.items()`` 保证），配合
    ``top_k`` 的稳定选择，使得**同样的输入一定得到同样的输出**。
    可复现是它作为真值的前提。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")

    query_values, query_norm = query  # type: ignore[misc]
    scores = [
        (
            metric.distance(query_values, query_norm, item.values, item.norm_sq),
            item.key,
        )
        for item in store.items()
    ]
    return ExactResult(hits=top_k(scores, k), metric=metric.name, k=k)


def recall_at_k(ann_keys: list[int], exact: ExactResult) -> float:
    """标准召回率：交集大小 / k。

    分母固定用 ``k`` 而不是 ``len(exact.keys)``：库里点数不足 k 时，
    后者会让召回率**虚高**（比如库里只有 3 个点、k=10，
    那 3 个都对上就是 3/3 = 100%，但"找 10 个"这个任务其实只完成了 3 个）。
    """
    if exact.k <= 0:
        return 0.0
    overlap = len(set(ann_keys) & set(exact.keys))
    return overlap / exact.k


def threshold_recall_at_k(
    ann_hits: list[tuple[float, int]],
    exact: ExactResult,
) -> float:
    """门槛召回率：ANN 结果里距离不超过第 k 名门槛的数量 / k。

    并列很多时用它来交叉验证标准召回率（见模块文档）。
    注意这里必须用**ANN 自己算出的距离**，而不是重新用精确实现算一遍 ——
    后者会掩盖"索引内部的距离计算有偏差"这类问题
    （比如 PQ 的近似距离系统性偏大），而那正是要测的东西。
    """
    if exact.k <= 0:
        return 0.0
    threshold = exact.threshold
    # 加一个极小容差：浮点下"相等的距离"可能差 1e-15，
    # 不加容差会把本该算命中的判成未命中，而且在并列集边缘随机发生。
    tolerance = 1e-9 * max(1.0, abs(threshold)) if threshold != float("inf") else 0.0
    hit = sum(1 for distance, _ in ann_hits if distance <= threshold + tolerance)
    # 除以 k 而不是除以 len(ann_hits)：结果数少于 k 时，缺的部分就是没找到。
    return min(hit, exact.k) / exact.k


def agreement(expected: ExactResult, other: ExactResult) -> float:
    """两次精确查询结果的一致度。用于自检与测试。

    在测试里断言"两次调用结果完全一致"是不可靠的（浮点顺序可能不同），
    断言"一致度 = 1.0"也一样。这里提供一个显式的口径，
    便于写出"应该完全一致"这种有意义的断言。
    """
    if not expected.hits and not other.hits:
        return 1.0
    if not expected.hits or not other.hits:
        return 0.0
    return len(set(expected.keys) & set(other.keys)) / max(len(expected.keys), 1)
