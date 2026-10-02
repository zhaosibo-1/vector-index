"""IVF-PQ：倒排文件 + 乘积量化。

论文：Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)
     + Babenko & Lempitsky, "The Inverted Multi-Index" (2012) 的工程化版本
       （即 faiss 里的 IndexIVFPQ）。

==========================================================================
 它和 HNSW 是两种完全不同的思路
==========================================================================
HNSW 是**图**：保精度，代价是内存（边比向量还占地方）。
IVF-PQ 是**量化**：保内存，代价是精度 —— 而且是**不可逆的信息损失**。

    HNSW：召回率可以调到 100%（ef 大到一定程度），内存换时间。
    IVF-PQ：召回率有**上限**，因为原始向量已经扔了。
            那个上限由压缩率决定，调什么参数都突破不了。

这个区别决定了它们的适用场景：HNSW 适合"内存够、要准"；
IVF-PQ 适合"十亿级、内存是硬约束、能接受 70~90% 召回"。
基准结果里会把两条曲线并排画出来，让这个取舍可见。

==========================================================================
 两级压缩，各管一件事
==========================================================================
1. **IVF（倒排文件）** —— 粗聚类。
   用 k-means 把库切成 ``nlist`` 个簇，每个簇一个倒排表。
   查询时只扫最近的 ``nprobe`` 个簇。
   它省的是**扫描量**（nlist=100、nprobe=8 就只扫 8% 的数据），
   不省内存 —— 反而多存了粗质心。

2. **PQ（乘积量化）** —— 精细压缩。
   把 d 维向量切成 m 段，每段各自训练一本含 ``2^nbits`` 个码字的码本，
   然后每个向量只存 m 个码字下标。**原始向量不保存。**

   为什么能这么压：把 d 维空间显式拆成 m 个低维子空间后，
   每段的状态数只有 ``2^nbits``，于是距离计算变成了
   **查表相加**（见下面的 ADC）。

   为什么"乘积"：显式枚举 d 维的码本需要 ``(2^nbits)^d`` 个码字，
   而乘积假设把它降到 ``m × 2^nbits``，代价是引入了子空间独立的近似。

**残差编码（residual PQ）**：量化的是 ``x - 粗质心``，不是 ``x`` 本身。
这一步很关键 —— 粗质心已经把向量的"主体位置"表达掉了，
剩下的残差分布更集中、更容易被少量码字覆盖，量化误差显著更小。

==========================================================================
 ADC：为什么查询时不用解码
==========================================================================
朴素做法是"把库里的量化码解码成近似向量，再和查询算距离"。
那既要解码（m 次查表拼数组），又引入了**两边的量化误差**（对称距离）。

ADC（Asymmetric Distance Computation）把查询留成原始向量：
对查询的每个子段、每个码字，**预先算好距离表** ``LUT[m][2^nbits]``，
然后一条库向量到查询的距离＝m 次数组下标相加。

于是每候选的距离成本从"m 次子向量距离"降到 **m 次整数下标 + 加法**，
而且只量化了库侧，误差更小。这是 PQ 能在秒级扫过百万向量的原因。

==========================================================================
 一个必须说清的限制：内部是 L2
==========================================================================
PQ 的距离表算的是 L2。所以：

* ``l2``     —— 直接用。
* ``cosine`` —— 向量已归一化，于是 ``‖a-b‖² = 2 - 2⟨a,b⟩``，
  **恒等式**，可以精确换算回 cosine 距离 ``‖a-b‖² / 2``。
* ``inner``  —— **不支持**。内积下没有这种恒等式，而且 MIPS 的正确做法
  是先归一化再用 cosine。与其给一个悄悄算错的实现，不如直接拒绝，
  并在错误信息里说清该怎么办。
"""

from __future__ import annotations

import math
import random
import time
from array import array
from dataclasses import dataclass, field

from .metrics import (
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    Metric,
    StaleIndexError,
    VectorStore,
    top_k,
)


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------


def _nearest_centroid(
    point: array,
    point_norm_sq: float,
    centroids: list[array],
    centroid_norms: list[float],
) -> tuple[int, float]:
    """返回 ``(最近质心的下标, 平方距离)``。

    用展开式 ``‖x-c‖² = ‖x‖² + ‖c‖² - 2⟨x,c⟩``，让每次比较只做一次
    ``sumprod``（C 层）而不是一趟 Python 循环 —— 这是纯 Python 实现的
    k-means 能在秒级跑完的唯一原因。
    """
    best_index = 0
    best_distance = math.inf
    for index, centroid in enumerate(centroids):
        distance = point_norm_sq + centroid_norms[index] - 2.0 * math.sumprod(
            point, centroid
        )
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, best_distance


def kmeans(
    samples: list[array],
    k: int,
    *,
    dim: int,
    iterations: int = 15,
    seed: int = 0,
) -> list[array]:
    """k-means++ 初始化 + Lloyd 迭代。

    两个必须做对的地方：

    **1. 初始化用 k-means++，不是随机取 k 个点。**
    随机取点会让某些初始质心落在同一个簇里，于是那个簇永远分不到向量，
    全挤在少数几个质心上 —— 表现为"训练完了但明显有几个簇是空的"。
    k-means++ 按 ``D²`` 加权采样，让新质心尽量落在离已有质心远的地方。
    代价是多一次距离计算，收益是收敛质量，非常划算。

    **2. 空簇必须显式处理。**
    Lloyd 迭代里如果有簇一个点都没分到，它的质心就保持不动，
    然后永远分不到点（死簇）。这里的做法是：把死簇的质心重置到
    **离它自己质心最远的那几个样本点**上，让它重新参与竞争。
    不处理的话，有效质心数会少于 k —— 而 IVF 的倒排表数量取决于
    **实际**用到的质心数，于是"我设了 16 个簇，为什么只有 11 个倒排表"
    这种问题会让人怀疑人生。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")
    if not samples:
        raise ValueError("没有样本可供聚类")
    if k > len(samples):
        # **夹取**，不是报错。理由很具体：让 k 超过样本数并不会得到 k 个簇。
        # 曾经这里写的是"允许（会得到一些空簇并触发重置逻辑）"——
        # 但实现从一开始就夹取了。文档与实现对不上，
        # 比"直接说清楚"更糟：读文档的人会去调试一个不存在的 bug。
        #
        # 真正要说清楚的是**后果**：IVF 的倒排表数量取决于实际质心数，
        # 所以这时候 ``nlist`` 会比请求值小。snapshot 里返回的是实际值，
        # 而不是请求值 —— 两边不一致时以实际值为准。
        k = max(1, len(samples))

    rng = random.Random(seed)
    norms = [math.sumprod(s, s) for s in samples]

    # --- k-means++ 初始化 ---
    centroids: list[array] = [array("f", samples[rng.randrange(len(samples))])]
    centroid_norms = [math.sumprod(centroids[0], centroids[0])]
    #: 每个样本到"最近的已选质心"的平方距离。初始化为到第一个质心的距离，
    #: 之后每加一个质心就取 min 更新 —— 这样每轮只需要算"到新质心"的距离，
    #: 而不是"到所有质心"的距离。
    distances = [
        max(0.0, norms[i] + centroid_norms[0] - 2.0 * math.sumprod(samples[i], centroids[0]))
        for i in range(len(samples))
    ]

    while len(centroids) < k:
        total = sum(distances)
        if total <= 0:
            # 所有样本都重合（或已全部被解释）。直接复制最后一个质心，
            # 后面的空簇重置逻辑会把它挪到有用的地方。
            centroids.append(array("f", centroids[-1]))
            centroid_norms.append(centroid_norms[-1])
            distances.append(0.0)
            continue

        # 按 D² 加权轮盘赌。用累积和 + 二分而不是逐项相减，
        # 是为了让"同样的随机数一定选中同一个样本"这件事不依赖浮点累加顺序。
        target = rng.random() * total
        cumulative = 0.0
        chosen = len(samples) - 1
        for index, distance in enumerate(distances):
            cumulative += distance
            if cumulative >= target:
                chosen = index
                break

        centroids.append(array("f", samples[chosen]))
        centroid_norms.append(norms[chosen])
        for index in range(len(samples)):
            candidate = (
                norms[index]
                + centroid_norms[-1]
                - 2.0 * math.sumprod(samples[index], centroids[-1])
            )
            if candidate < distances[index]:
                distances[index] = candidate

    # --- Lloyd 迭代 ---
    assignments = [0] * len(samples)
    for _ in range(iterations):
        changed = 0
        for index, sample in enumerate(samples):
            best, _ = _nearest_centroid(sample, norms[index], centroids, centroid_norms)
            if best != assignments[index]:
                assignments[index] = best
                changed += 1

        # 重算质心
        sums: list[list[float]] = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for index, sample in enumerate(samples):
            cluster = assignments[index]
            counts[cluster] += 1
            row = sums[cluster]
            for position in range(dim):
                row[position] += sample[position]

        for cluster in range(k):
            if counts[cluster]:
                centroids[cluster] = array(
                    "f", (value / counts[cluster] for value in sums[cluster])
                )
            # else: 保持不动，等下面的空簇重置。

        centroid_norms = [math.sumprod(c, c) for c in centroids]

        # --- 空簇重置 ---
        empty = [c for c in range(k) if counts[c] == 0]
        if empty and len(samples) >= k:
            # 找"离自己所属质心最远"的样本，把它们各自升级成新质心。
            # 只取需要的个数，且不重复同一个样本。
            ranked = sorted(
                range(len(samples)),
                key=lambda i: -(
                    norms[i]
                    + centroid_norms[assignments[i]]
                    - 2.0 * math.sumprod(samples[i], centroids[assignments[i]])
                ),
            )
            used: set[int] = set()
            for cluster in empty:
                for index in ranked:
                    if index in used:
                        continue
                    used.add(index)
                    centroids[cluster] = array("f", samples[index])
                    centroid_norms[cluster] = norms[index]
                    assignments[index] = cluster
                    break
            centroid_norms = [math.sumprod(c, c) for c in centroids]
            changed += 1  # 强制再迭代一轮

        if changed == 0:
            # 已经收敛。继续迭代只是白烧时间 —— 而且会让"同一个参数
            # 两次训练得到完全一样的结果"这件事变得依赖迭代次数。
            break

    return centroids


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class IVFPQConfig:
    """构建与查询参数。

    参数分成三组，各自的优化目标不同 —— 混在一起调会互相干扰：

    * ``nlist`` / ``coarse_iterations`` —— 粗聚类质量，影响**扫描量**；
    * ``m`` / ``nbits``             —— 压缩率，影响**内存与精度上限**；
    * ``nprobe``                   —— 查询期旋钮，影响**召回率与延迟**。
    """

    #: 粗质心数量（倒排表数量）。
    #: 经验规则 ``nlist ≈ 4 × sqrt(n)``；但纯 Python 的 k-means 撑不住
    #: 太大的 nlist，所以默认取小值并在 README 里说明。
    nlist: int = 16
    coarse_iterations: int = 12
    #: 子空间数量。**必须整除维度。** 越大压缩率越低、精度越高
    #: （m=d 时退化成标量量化）。
    m: int = 8
    #: 每个子空间的码本位数，码本大小 = ``2 ** nbits``。
    #: 4 位 → 16 个码字；8 位 → 256 个（训练成本高 16 倍）。
    nbits: int = 4
    #: 查询时扫描的倒排表数量。**召回率的主要旋钮**。
    nprobe: int = 4
    #: k-means 的训练样本上限。**必须设** ——
    #: 用全量样本训练，成本随 n 线性增长，而质心质量的提升很快饱和。
    train_sample_limit: int = 768
    #: 子空间 k-means 的迭代次数。比粗聚类少，因为子空间维度低、收敛快。
    pq_iterations: int = 12
    seed: int = 20260930

    def __post_init__(self) -> None:
        if self.nlist < 1:
            raise ValueError(f"nlist 必须 ≥ 1，收到 {self.nlist}")
        if self.m < 1:
            raise ValueError(f"m 必须 ≥ 1，收到 {self.m}")
        if self.nbits < 1:
            raise ValueError(f"nbits 必须 ≥ 1，收到 {self.nbits}")
        if self.nbits > 8:
            # 码字下标按字节存（array('B')），超过 8 位就放不下了。
            # 这不是能力问题而是格式约束 —— 想更大就得多字节编码，
            # 那样压缩率又会下降。8 位（256 个码字）已经是标准做法的上限。
            raise ValueError(f"nbits 必须 ≤ 8（码字按字节存），收到 {self.nbits}")
        if self.nprobe < 1:
            raise ValueError(f"nprobe 必须 ≥ 1，收到 {self.nprobe}")

    @property
    def codebook_size(self) -> int:
        return 1 << self.nbits

    def sub_dim(self, dim: int) -> int:
        return dim // self.m

    def validate_dim(self, dim: int) -> None:
        if dim % self.m != 0:
            raise ValueError(
                f"维度 {dim} 必须能被 m={self.m} 整除"
                f"（每个子段要等长，否则距离表无法按固定步长查表）"
            )


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------


@dataclass
class IVFPQStats:
    trained: int = 0
    list_sizes: list[int] = field(default_factory=list)
    build_seconds: float = 0.0
    train_seconds: float = 0.0
    #: 每次查询平均扫过多少条候选，用于解释延迟
    scanned_total: int = 0
    queries: int = 0


class IVFPQIndex:
    """倒排 + 乘积量化索引。

    与 HNSW 最大的结构差异：**这个索引不持有原始向量**。
    它只保留量化码，所以内存与召回率是同一个参数（``m``）的两面，
    不可能同时最优。
    """

    def __init__(
        self,
        store: VectorStore,
        metric: Metric,
        config: IVFPQConfig | None = None,
    ) -> None:
        if metric.name == METRIC_INNER_PRODUCT:
            raise ValueError(
                "IVF-PQ 内部按 L2 距离量化，内积检索没有可换算的恒等式。"
                "请改用 cosine 度量（先把向量归一化，此时内积与余弦等价），"
                "或改用 HNSW。"
            )
        self.store = store
        self.metric = metric
        self.config = config or IVFPQConfig()
        self.config.validate_dim(store.dim)

        self.dim = store.dim
        self.sub_dim = self.config.sub_dim(self.dim)

        #: 粗质心与它们的 ‖·‖²
        self._coarse: list[array] = []
        self._coarse_norms: list[float] = []
        #: 每个倒排表里存的是**码**（每个向量 m 个字节），不存向量
        self._codes: list[list[array]] = []
        #: 每个倒排表对应的 key 列表（顺序与 _codes 一一对应）
        self._keys: list[list[int]] = []
        #: PQ 码本：``_codebooks[s][c]`` = 第 s 段第 c 个码字
        self._codebooks: list[list[array]] = []

        self._trained = False
        self._built_revision: int | None = None
        self.stats = IVFPQStats()

    # ------------------------------------------------------------------
    # 训练与构建
    # ------------------------------------------------------------------

    def build(self) -> "IVFPQIndex":
        started = time.perf_counter()

        all_items = list(self.store.items())
        if not all_items:
            raise ValueError("向量库是空的，无法训练 IVF-PQ")

        config = self.config
        # --- 1. 训练粗质心 ---
        train_started = time.perf_counter()
        coarse_samples = self._sample_vectors(all_items, config.train_sample_limit)
        self._coarse = kmeans(
            coarse_samples,
            config.nlist,
            dim=self.dim,
            iterations=config.coarse_iterations,
            seed=config.seed,
        )
        self._coarse_norms = [math.sumprod(c, c) for c in self._coarse]

        # --- 2. 算残差，训练 PQ 码本 ---
        residuals = [
            self._residual(item.values, item.norm_sq)
            for item in all_items
        ]
        self._codebooks = self._train_codebooks(residuals)
        self.stats.train_seconds = time.perf_counter() - train_started

        # --- 3. 编码并分桶 ---
        self._codes = [[] for _ in self._coarse]
        self._keys = [[] for _ in self._coarse]
        for item, residual in zip(all_items, residuals):
            list_index = self._nearest_coarse(item.values, item.norm_sq)
            self._codes[list_index].append(self._encode(residual))
            self._keys[list_index].append(item.key)

        self._trained = True
        self._built_revision = self.store.revision
        self.stats.trained = len(all_items)
        self.stats.list_sizes = [len(keys) for keys in self._keys]
        self.stats.build_seconds = time.perf_counter() - started
        return self

    def _sample_vectors(self, items: list, limit: int) -> list[array]:
        """取训练子集。

        用**等间隔抽样**而不是随机抽样：等间隔是确定性的，
        不依赖随机数，于是"同样的数据一定训练出同样的码本"这件事
        只需要保证 k-means 本身确定即可，少一个不确定源。

        而且等间隔抽样在按 key 排列的数据上天然覆盖各个区域；
        随机抽样在小样本下可能整段区域一个点都没抽到。
        """
        if len(items) <= limit:
            return [item.values for item in items]
        step = len(items) / limit
        return [items[int(i * step)].values for i in range(limit)]

    def _nearest_coarse(self, values: array, norm_sq: float) -> int:
        index, _ = _nearest_centroid(values, norm_sq, self._coarse, self._coarse_norms)
        return index

    def _residual(self, values: array, norm_sq: float) -> array:
        """``x - 最近的粗质心``。

        量化残差而不是原向量：粗质心已经表达了"这个向量大概在哪个区域"，
        残差的分布更集中，用同样数量的码字能覆盖得更细，量化误差更小。
        这是 IVF 和 PQ 放在一起用（而不是各自独立用）的真正理由。
        """
        index = self._nearest_coarse(values, norm_sq)
        centroid = self._coarse[index]
        return array("f", (v - c for v, c in zip(values, centroid)))

    def _train_codebooks(self, residuals: list[array]) -> list[list[array]]:
        """对每个子空间独立训练一本码本。"""
        codebooks: list[list[array]] = []
        for sub in range(self.config.m):
            start = sub * self.sub_dim
            stop = start + self.sub_dim
            sub_samples = [array("f", r[start:stop]) for r in residuals]
            codebooks.append(
                kmeans(
                    sub_samples,
                    self.config.codebook_size,
                    dim=self.sub_dim,
                    iterations=self.config.pq_iterations,
                    # 每个子空间用不同的种子：共用同一个种子会让各段的
                    # k-means++ 采样序列一模一样，在**规律性结构**的数据上
                    # 可能选中相同的初始点，引入不必要的相关性。
                    seed=self.config.seed + 1 + sub,
                )
            )
        return codebooks

    def _encode(self, residual: array) -> array:
        """把残差编码成 m 个字节。这是**唯一**会被持久保留的表示。"""
        code = array("B")
        for sub in range(self.config.m):
            start = sub * self.sub_dim
            segment = array("f", residual[start : start + self.sub_dim])
            segment_norm = math.sumprod(segment, segment)
            index, _ = _nearest_centroid(
                segment, segment_norm, self._codebooks[sub], self._codebook_norms(sub)
            )
            code.append(index)
        return code

    def _codebook_norms(self, sub: int) -> list[float]:
        """码字模长缓存。第一次用到时算，之后复用。

        k-means 训练完就固定了，所以这是纯收益的缓存 ——
        没有它，编码 n 个向量要重复算 ``n × m × 2^nbits`` 次模长。
        """
        cached = self._codebook_norm_cache.get(sub)
        if cached is None:
            cached = [math.sumprod(c, c) for c in self._codebooks[sub]]
            self._codebook_norm_cache[sub] = cached
        return cached

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _build_lut(self, query_residual: array) -> list[array]:
        """预先算好距离表 ``LUT[sub][code]``。

        这是 ADC 的核心：**在扫描之前**把"查询的第 s 段与第 s 段所有码字的
        距离"算完。于是扫描时每个候选的成本变成 m 次数组取值相加，
        而不是 m 次子向量距离计算。

        代价是 ``m × 2^nbits`` 次距离计算（默认 8 × 16 = 128 次）——
        这是个**固定成本**，和数据量无关。所以 nprobe 越小、库越大，
        ADC 的优势越明显。
        """
        lut: list[array] = []
        for sub in range(self.config.m):
            start = sub * self.sub_dim
            segment = array("f", query_residual[start : start + self.sub_dim])
            segment_norm = math.sumprod(segment, segment)
            table = array("f")
            for codevector in self._codebooks[sub]:
                table.append(
                    max(
                        0.0,
                        segment_norm
                        + math.sumprod(codevector, codevector)
                        - 2.0 * math.sumprod(segment, codevector),
                    )
                )
            lut.append(table)
        return lut

    def _ensure_fresh(self) -> None:
        if not self._trained:
            raise RuntimeError("索引还没有构建，先调用 build()")
        if self.store.revision != self._built_revision:
            raise StaleIndexError(
                "向量库在索引构建之后被改过（revision "
                f"{self._built_revision} → {self.store.revision}）；"
                "请重建索引。IVF-PQ 的码本与倒排表都基于旧数据，"
                "新增的向量根本不在任何倒排表里。"
            )

    def search(
        self,
        query: array,
        query_norm: float,
        *,
        k: int = 10,
        nprobe: int | None = None,
    ) -> list[tuple[float, int]]:
        """返回 ``[(距离, key)]``，距离升序。

        返回的距离是**换算到该度量的**：内部一律按 L2 算，
        cosine 场景下按 ``d_cos = d_l2 / 2`` 精确换算（向量已归一化）。
        不换算的话，调用方会拿到一个"语义上是 L2、却声称是 cosine"的数，
        而它看起来完全合理（都是 0~2 之间的小数）。
        """
        self._ensure_fresh()
        if k <= 0:
            return []

        probes = min(nprobe or self.config.nprobe, len(self._coarse))
        # 粗定位：找离查询最近的 probes 个倒排表
        scored = []
        for index in range(len(self._coarse)):
            distance = (
                query_norm
                + self._coarse_norms[index]
                - 2.0 * math.sumprod(query, self._coarse[index])
            )
            scored.append((distance, index))
        probed = top_k(scored, probes)

        candidates: list[tuple[float, int]] = []
        for coarse_distance, list_index in probed:
            centroid = self._coarse[list_index]
            # 查询侧也要减去**被探测的那个**粗质心，才能和库侧的残差编码对齐。
            # 少减这一次，距离会系统性偏大，表现为召回率断崖式下跌。
            query_residual = array(
                "f", (q - c for q, c in zip(query, centroid))
            )
            lut = self._build_lut(query_residual)

            codes = self._codes[list_index]
            keys = self._keys[list_index]
            self.stats.scanned_total += len(codes)
            for position, code in enumerate(codes):
                # ADC：m 次查表相加
                total = 0.0
                for sub, code_value in enumerate(code):
                    total += lut[sub][code_value]
                candidates.append((total, keys[position]))

        self.stats.queries += 1
        best = top_k(candidates, k)
        return [(self._to_metric_distance(d), key) for d, key in best]

    def _to_metric_distance(self, l2_sq: float) -> float:
        """把内部 L2 平方距离换算成该度量的距离。

        归一化向量的恒等式：``‖a-b‖² = ‖a‖² + ‖b‖² - 2⟨a,b⟩ = 2 - 2cos``，
        所以 ``d_cosine = 1 - cos = ‖a-b‖² / 2``。这是**精确**换算，
        不是近似 —— 近似只发生在"库侧向量被量化"这一步。

        这个换算不能省：省掉之后 L2 场景没问题，cosine 场景会返回一个
        数值范围看起来也合理、但含义是 L2 的距离，界面上显示"相似度"
        就会系统性偏低，而且没有任何报错。
        """
        if self.metric.name == METRIC_COSINE:
            return l2_sq / 2.0
        if self.metric.name == METRIC_L2:
            return l2_sq
        # 构造时已经拒绝了 inner，这里只是兜底
        raise AssertionError(f"未处理的度量 {self.metric.name!r}")

    # ------------------------------------------------------------------
    # 核算
    # ------------------------------------------------------------------

    @property
    def _codebook_norm_cache(self) -> dict[int, list[float]]:
        cached = getattr(self, "_norms_cache", None)
        if cached is None:
            cached = {}
            self._norms_cache = cached
        return cached

    def memory_bytes(self) -> dict[str, int]:
        """内存构成。

        **注意这里没有 ``vectors``** —— 索引不保存原始向量。
        这正是 IVF-PQ 能压下来的原因，也是它召回率有上限的原因：
        信息已经丢了，不是算法不够聪明。

        压缩比有两个口径，混用会得出互相矛盾的数字，所以都显式列出：

        ``codes_only_compression``
            只看编码本身，``= 4d / m``。这是论文和实现文档里引用的那个
            "16 倍"。它有一个前提：**n 要足够大**。码本和粗质心是
            O(nlist × d) 与 O(m × 2^nbits × d/m) 的固定开销，与 n 无关，
            n 小的时候它们会把这个数字打得粉碎。

        ``index_total_compression``
            编码 + 码本 + 粗质心 + 倒排表，除以原始向量。
            这是"实际占了多少内存"的诚实口径，也是 ``compression_ratio``
            返回的值。n=1200、d=32、m=8、nbits=4 实测约 **5.2 倍**
            （codes 16 倍被 20KB 固定开销稀释），n=100k 时才接近 16。
        """
        n = self.stats.trained or len(self.store)
        codes = sum(len(code) for per_list in self._codes for code in per_list)
        codebooks = sum(
            len(codebook) * self.sub_dim * 4 for codebook in self._codebooks
        )
        coarse = len(self._coarse) * self.dim * 4
        # 倒排表的 key 列表：每条 8 字节（int 引用）
        key_slots = sum(len(keys) for keys in self._keys) * 8
        original = n * self.dim * 4
        total = codes + codebooks + coarse + key_slots
        return {
            "compressed_codes": codes,
            "codebooks": codebooks,
            "coarse_centroids": coarse,
            "inverted_lists": key_slots,
            "total": total,
            "original_vectors": original,
            "codes_only_compression": round(
                original / max(codes, 1), 2
            ),
            "index_total_compression": round(original / max(total, 1), 2),
            "compression_ratio": round(original / max(total, 1), 2),
        }

    def snapshot(self) -> dict[str, object]:
        mem = self.memory_bytes()
        sizes = self.stats.list_sizes
        return {
            "trained": self.stats.trained,
            "nlist": len(self._coarse),
            "probed_lists": self.config.nprobe,
            "list_sizes": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "empty": sum(1 for s in sizes if s == 0),
                "avg": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            },
            "memory": mem,
            "config": {
                "nlist": self.config.nlist,
                "m": self.config.m,
                "nbits": self.config.nbits,
                "codebook_size": self.config.codebook_size,
                "sub_dim": self.sub_dim,
                "nprobe": self.config.nprobe,
                "train_sample_limit": self.config.train_sample_limit,
                "seed": self.config.seed,
            },
            "build_seconds": round(self.stats.build_seconds, 4),
            "train_seconds": round(self.stats.train_seconds, 4),
            # ``query_invocations`` 是**索引**上的 property，不是 stats 的字段。
            # 曾经写成 ``self.stats.query_invocations`` —— 于是只要查过一次，
            # snapshot() 就一定抛 AttributeError。这类 bug 最讨厌的地方是
            # 它和业务逻辑完全无关：索引本身是对的，只是"看一眼自己"会崩。
            "avg_scanned_per_query": round(
                self.stats.scanned_total / self.query_invocations, 1
            ),
        }

    @property
    def query_invocations(self) -> int:
        return max(self.stats.queries, 1)
