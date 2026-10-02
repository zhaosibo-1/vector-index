"""距离度量与向量容器：整个项目里所有"远近"判断的唯一来源。

==========================================================================
 第一条约定：一切距离都是「越小越近」
==========================================================================
三种度量对外看起来差别很大，但在这里全部归一成**最小化距离**：

    L2        distance = ‖a-b‖²
    cosine    distance = 1 - cos(a, b)          （范围 [0, 2]）
    inner     distance = -⟨a, b⟩                （内积越大越近，所以取负）

为什么必须统一：图构造、候选剪枝、PQ 查表、top-k 堆，这些代码都假设
「距离小的优先」。如果让 L2 走一套（最小化）、内积走另一套（最大化），
那么每一处比较、每一个堆的方向都要分情况写 —— 而这类代码出错的
表现是**结果不对但看起来正常**（返回了最不相似的 10 个），
非常难从输出里看出来。统一在入口处转换一次，后面就只有一个方向。

==========================================================================
 第二条约定：距离可以"降级"成只保证序关系
==========================================================================
`distance()` 返回的是"能拿来排序的值"，不保证是严格的数学距离。
L2 走的是展开式 ‖a-b‖² = ‖a‖² + ‖b‖² - 2⟨a,b⟩，这是**恒等式**，
但在浮点下会有一点误差（‖a‖² 很大、2⟨a,b⟩ 也很大的时候，相减可能丢掉
低位有效数字），所以理论上可能出现极小的负数。

处理方式：**不夹取到 0**。夹取看起来更"正确"，但会把大量"其实有细微差别"
的候选压成同一个 0，反而破坏排序 —— 而这些候选在 tie-break 时顺序会不稳定，
导致同样的输入两次运行得到不同的结果。留下那个 -1e-7 完全无害。

==========================================================================
 第三条约定：模长预先算好
==========================================================================
L2 的展开式和 cosine 都要用到 ‖v‖²。如果每次比较都现算，等于把
工作量翻倍；而索引里同一个库向量会被比较成千上万次。
所以 `VectorStore` 在**写入时**算一次并存下来。

代价是"改了向量却忘了更新模长"，于是提供 `replace()` 而不是允许直接改 ——
接口层面就不给弄错的机会。
"""

from __future__ import annotations

import math
import operator
from array import array
from dataclasses import dataclass
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# sumprod：Python 3.12+ 有 C 实现的 math.sumprod，更早版本回落到 map/mul
# ---------------------------------------------------------------------------
# 实测（32 维、20000 次比较，本机 Python 3.13）：
#     for + range          4.46 µs / 次
#     for + zip            3.33 µs / 次
#     math.sumprod 展开     1.54 µs / 次     ← 比 for+range 快约 2.9 倍
#
# 这个差距在索引里会被放大很多倍：一次 HNSW 构建要做几十万到几百万次
# 距离计算，1.5µs 与 4.5µs 的差别就是"几秒"和"十几秒"的区别。
#
# 为什么要写回落分支而不是直接要求 Python ≥ 3.12：
# CI 跑 3.11 与 3.12 两档；而 3.11 上完全没有这个函数，
# 不做回落就是 ImportError，而不是"慢一点"。
try:  # pragma: no cover - 分支取决于解释器版本
    _sumprod = math.sumprod
    SUMPROD_IS_NATIVE = True
except AttributeError:  # pragma: no cover - Python < 3.12
    def _sumprod(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(map(operator.mul, a, b))

    SUMPROD_IS_NATIVE = False

#: 浮点比较的容差。用来判断"两个距离是不是一样近"，
#: 而不是用来判断"相等"—— 向量距离几乎没有精确相等的情况。
EPSILON = 1e-12


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """内积。"""
    return _sumprod(a, b)


def squared_norm(v: Sequence[float]) -> float:
    """‖v‖²（不取平方根，省一次开方）。"""
    return _sumprod(v, v)


# ---------------------------------------------------------------------------
# 度量
# ---------------------------------------------------------------------------

#: 支持的度量名。对外接口只接受这三个字符串，不接受任意可调用对象 ——
#: 索引需要知道度量的**性质**（是否需要归一化、是否可以用内积加速），
#: 传一个黑盒函数进来会让那些优化全部失效。
METRIC_L2 = "l2"
METRIC_COSINE = "cosine"
METRIC_INNER_PRODUCT = "inner"
VALID_METRICS: tuple[str, ...] = (METRIC_L2, METRIC_COSINE, METRIC_INNER_PRODUCT)


@dataclass(frozen=True)
class Metric:
    """一个度量及其行为约定。"""

    name: str

    def __post_init__(self) -> None:
        if self.name not in VALID_METRICS:
            raise ValueError(
                f"未知的度量 {self.name!r}；合法值：{'、'.join(VALID_METRICS)}"
            )

    # -- 性质 -------------------------------------------------------------

    @property
    def needs_normalization(self) -> bool:
        """写入时是否要把向量归一化成单位向量。

        cosine 归一化之后 cos = ⟨a,b⟩，少一次除法与开方。
        这一步是**写入时的一次性成本**换**每次查询的持续收益** ——
        归一化在索引里是正确的，因为 cosine 本身对模长不敏感。
        """
        return self.name == METRIC_COSINE

    @property
    def is_similarity(self) -> bool:
        """语义上是不是"越大越相似"。

        只用于**展示**（界面上说"相似度 0.93"比说"距离 0.07"直观）。
        内部一律用 distance，不做任何分支。
        """
        return self.name in (METRIC_COSINE, METRIC_INNER_PRODUCT)

    # -- 距离 -------------------------------------------------------------

    def distance(
        self,
        a: Sequence[float],
        norm_a: float,
        b: Sequence[float],
        norm_b: float,
    ) -> float:
        """返回可排序的距离值，**越小越近**。

        ``norm_a`` / ``norm_b`` 是两侧的 ‖·‖²（见 `squared_norm`）。
        调用方负责传对 —— 这是性能与正确性之间的取舍：
        在函数里现算就等于把这个优化白做了。
        """
        product = _sumprod(a, b)
        if self.name == METRIC_L2:
            # 展开式：省掉逐元素相减与平方
            return norm_a + norm_b - 2.0 * product
        if self.name == METRIC_COSINE:
            # 向量已归一化，norm 都是 1，但保留通用写法以容忍浮点误差
            denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
            if denominator < EPSILON:
                # 零向量与任何向量的余弦没有定义。返回最大距离（最不相似）
                # 而不是抛异常：数据集里混进一个零向量不该让整个查询失败。
                return 2.0
            return 1.0 - product / denominator
        return -product

    def distance_to_vector(
        self, a: Sequence[float], b: "StoredVector"
    ) -> float:
        """对索引内部存的向量（自带模长）求距离，省掉调用方查模长。"""
        return self.distance(a, _sumprod(a, a), b.values, b.norm_sq)

    def similarity_of(self, distance: float) -> float:
        """把距离翻译成人类可读的相似度，仅用于展示。"""
        if self.name == METRIC_COSINE:
            return 1.0 - distance
        if self.name == METRIC_INNER_PRODUCT:
            return -distance
        # L2 没有天然的"相似度"上限（距离无界），用一个单调递减的映射
        # 让它落在 (0, 1] 里方便展示。**它不是概率，也不是余弦**，
        # 界面上必须标出这是"按 L2 距离换算的相对相似度"。
        return 1.0 / (1.0 + max(distance, 0.0))


def make_metric(name: str) -> Metric:
    return Metric(name)


# ---------------------------------------------------------------------------
# 向量容器
# ---------------------------------------------------------------------------


class StaleIndexError(RuntimeError):
    """向量库在某个索引构建之后被改过。

    定义在 metrics 层而不是某个索引里，是因为它属于"索引契约"：
    HNSW 和 IVF-PQ 都会遇到它。如果各自抛自己的异常，
    调用方要么写两个 except，要么漏掉一个 ——
    而漏掉的那一类恰好是最难复现的。

    继承 ``RuntimeError`` 而不是 ``Exception``：这是**程序缺陷**
    （该重建索引却没重建），不是用户输入错误，
    但也不该用一个全新的基类把调用方的捕获逻辑全打乱。
    """


@dataclass(frozen=True)
class StoredVector:
    """一条已入库的向量。

    ``values`` 用 ``array('f')`` 而不是 ``list``：
    32 维 float32 的 ``array`` 占 128 字节，等价的 ``list`` 在 64 位 CPython 上
    光指针数组就要 8×32 = 256 字节，再算上 32 个被引用的 float 对象
    （每个约 24 字节）——**约 5 倍**。内存恰好是这个项目要量化的指标，
    用 list 会让这个数字失去意义。
    """

    key: int
    values: array
    norm_sq: float
    #: 可选的外部标识（比如文档 id），原样存回给调用方
    label: str = ""


class VectorStore:
    """按整数 key 存向量，写入时算好模长。

    刻意**不提供直接修改向量的方法**：改了向量却忘了重算 ``norm_sq``
    会让所有距离都悄悄算错，而且错得没有任何迹象（只是"结果有点不对"）。
    要换向量就 ``replace()``，它会把模长一起更新。
    """

    def __init__(self, dim: int, metric: Metric) -> None:
        if dim <= 0:
            raise ValueError(f"维度必须是正整数，收到 {dim}")
        self.dim = dim
        self.metric = metric
        self._vectors: dict[int, StoredVector] = {}
        self._next_key = 0
        #: 每次写入（add / replace）都 +1。
        #:
        #: 存在的唯一目的是让索引能发现"向量库已经变了"。
        #: HNSW / IVF-PQ 都是在某个时刻对**当时的**向量库建的图/倒排表；
        #: 之后库里新增或替换了向量，索引就过期了。过期索引不会报错 ——
        #: 它会安安静静地返回基于旧图的、越来越差的最近邻。
        #: 这类问题最典型的症状是"召回率随时间莫名下降"，
        #: 而排查时几乎不会有人想到去怀疑索引版本。
        self.revision = 0

    # -- 写入 -------------------------------------------------------------

    def _prepare(self, raw: Iterable[float]) -> array:
        values = array("f", raw)
        if len(values) != self.dim:
            raise ValueError(
                f"维度不匹配：期望 {self.dim}，收到 {len(values)}"
            )
        if not all(math.isfinite(v) for v in values):
            # 一条 inf/nan 会让所有用到它的距离变成 nan，
            # 而 nan 的比较恒为 False —— 于是这个点在图上"谁都连不上"，
            # 表现为召回率莫名其妙地掉，且找不到原因。在入口就拦住。
            raise ValueError("向量里出现了 inf 或 nan，无法入索引")
        if self.metric.needs_normalization:
            norm = math.sqrt(_sumprod(values, values))
            if norm < EPSILON:
                raise ValueError("零向量无法做余弦归一化")
            values = array("f", (v / norm for v in values))
        return values

    def add(self, raw: Iterable[float], label: str = "") -> int:
        values = self._prepare(raw)
        key = self._next_key
        self._next_key += 1
        self._vectors[key] = StoredVector(
            key=key,
            values=values,
            norm_sq=_sumprod(values, values),
            label=label,
        )
        self.revision += 1
        return key

    def replace(self, key: int, raw: Iterable[float], label: str | None = None) -> None:
        """整条替换（含模长重算）。这是唯一的修改入口。"""
        if key not in self._vectors:
            raise KeyError(f"没有 key={key} 的向量")
        values = self._prepare(raw)
        old = self._vectors[key]
        self._vectors[key] = StoredVector(
            key=key,
            values=values,
            norm_sq=_sumprod(values, values),
            label=old.label if label is None else label,
        )
        self.revision += 1

    def prepare_query(self, raw: Iterable[float]) -> tuple[array, float]:
        """把外部查询向量处理成内部格式，返回 ``(values, norm_sq)``。

        查询向量走和入库向量同一条预处理（含归一化）——
        两边不一致是这一类 bug 的经典来源：库里的向量归一化过、
        查询的没归一化，cosine 就会算出大于 1 的荒谬值。
        """
        values = self._prepare(raw)
        return values, _sumprod(values, values)

    # -- 读取 -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: object) -> bool:
        return key in self._vectors

    def get(self, key: int) -> StoredVector:
        return self._vectors[key]

    def all_keys(self) -> list[int]:
        return sorted(self._vectors)

    def values_of(self, key: int) -> array:
        return self._vectors[key].values

    def label_of(self, key: int) -> str:
        return self._vectors[key].label

    def items(self) -> Iterable[StoredVector]:
        """按 key 升序迭代。**必须是稳定顺序** —— 索引构建的可复现性
        依赖这一点，用 dict 的插入顺序虽然当前也稳定，但显式排序更保险。"""
        for key in sorted(self._vectors):
            yield self._vectors[key]

    # -- 内存 -------------------------------------------------------------

    def memory_bytes(self) -> int:
        """向量本体的内存占用。

        只算 ``array`` 的缓冲区（``.itemsize × 长度``），
        不算 Python 对象头与 dict 开销。

        为什么这么定义：这个项目的对比对象是「全精度向量 vs 压缩编码」，
        要量的是**数据本身的体积**。把对象头算进来会让"每向量 24 字节的
        固定开销"盖过 d=16 时压缩带来的差异（16 维 float32 = 64 字节，
        加上对象头就变成 88，压缩到 4 字节的对比被稀释了）。
        界面上会同时给出「数据体积」与「进程内存」两个数，口径分开写清楚。
        """
        return sum(v.values.itemsize * len(v.values) for v in self._vectors.values())

    def stats(self) -> dict[str, object]:
        return {
            "count": len(self._vectors),
            "dim": self.dim,
            "metric": self.metric.name,
            "dtype": "float32",
            "vector_bytes": self.memory_bytes(),
            "bytes_per_vector": (
                self.memory_bytes() // len(self._vectors) if self._vectors else 0
            ),
        }


# ---------------------------------------------------------------------------
# top-k 选择
# ---------------------------------------------------------------------------


def top_k(scores: Iterable[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    """从 ``(距离, key)`` 里取最小的 k 个，按距离升序返回。

    用 `heapq.nsmallest` 而不是 `sorted(...)[:k]`：k 通常远小于候选数
    （k=10，候选几千），全排序是 O(n log n)，而 nsmallest 是 O(n log k)。

    平局时的顺序：nsmallest 是稳定的（保持输入顺序），
    而输入顺序由调用方决定 —— 索引内部一律按 key 升序遍历候选，
    所以同样的输入一定给出同样的输出。**结果可复现**比"哪个并列项排前面"
    重要得多：不可复现的输出会让测试变成"偶尔失败"。
    """
    import heapq

    if k == 0:
        return []
    if k < 0:
        # k=0 返回空是合理的调用（"我什么都不要"），
        # 但负 k 只可能是调用方算错了下标 —— 静默返回空列表会让
        # "上游 range 算错"这种 bug 一路绿灯通过，最后表现为
        # "结果莫名少了几条"，而没有人会怀疑到这里。
        raise ValueError(f"k 不能为负，收到 {k}")
    return heapq.nsmallest(k, scores, key=lambda item: item[0])
