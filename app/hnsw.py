"""HNSW：分层可导航小世界图（Hierarchical Navigable Small World）。

论文：Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs" (2016/2018)。

==========================================================================
 它为什么长成"分层"这个样子
==========================================================================
先说清要解决的问题。在图上做贪心搜索做最近邻，会遇到一个矛盾：

* 想让贪心"走得快"，每个节点应该只有很少的邻居（步数少、每次比较便宜）；
* 想让贪心"不卡在局部最优"，每个节点又应该有很多长程连接（否则从
  入口走到目标附近要绕很多步，甚至在簇边界上直接走不出去）。

单层图没法同时满足。HNSW 的解法是**分层**：

* 第 0 层含全部节点，边稠密、跨度小 —— 负责"精确落点"；
* 每往上一层，节点数按几何分布骤减（``P(level ≥ l) = exp(-l / mL)``），
  边跨度大 —— 负责"快速粗定位"；
* 查询从最高层的唯一入口开始，**每层只做贪心（ef=1）逼近**，
  然后下探一层，把上层的落点当作下层的入口。

这等价于在图上模拟了"跳表"：用 O(log n) 层的期望高度换来
O(log n) 的搜索复杂度，同时保留第 0 层的稠密结构保证精度。

关键洞察：**构造时的"贪心下降"和查询时的完全一致**。
所以插入一个新节点时不需要任何预处理 —— 边查边建，
这也是 HNSW 相比"需要离线训练的树/量化方法"最大的工程优势。

==========================================================================
 三个必须做对的细节（做错了会静默劣化，不报错）
==========================================================================

1. **邻居选择要用"多样性启发式"，不能只取最近的 M 个。**
   只取最近 M 个的话，如果查询点落在一个簇的边缘，它的 M 个最近邻
   可能**全都在同一个方向**，于是图上没有任何一条边指向"另一边"。
   一旦贪心走进这个方向就再也出不来 —— 表现为召回率对某些查询
   极差（不是普遍差），而且加多少 ef 都救不回来。

   启发式的规则：候选 c 只在"c 到查询点的距离 < c 到任何已选邻居的距离"
   时才被保留。直觉是"每个方向最多留一个代表"。

2. **双向边必须剪枝，而且剪枝也要用启发式。**
   给新节点连边时，被连的一方（旧节点）的邻居数会超出上限 M_max。
   直接删掉最远的那几个是错的 —— 那会系统性偏向"保留同一方向的边"，
   把上一条的多样性重新破坏掉。剪枝必须复用同一个启发式。

3. **熔断式的早停条件。**
   搜索候选堆里的最近距离如果已经大于当前结果集里最差的，就可以停。
   不加这一条会让搜索退化成"把整个连通分量走一遍"—— 结果一样，
   但耗时差一个数量级，而且 ef 越大越明显。

==========================================================================
 内存去哪了
==========================================================================
这个索引的内存里，**边是一项和向量本体同量级的开销**，不能忽略：

* 向量本体：``n × d × 4`` 字节（float32）；
* 边：每个节点平均 ``M0 + M × (层数-1)`` 条边，
  每条边在 Python 里是一个 list 槽位（8 字节指针），指向一个共享的
  int 对象。第 0 层的 M0 = 2M 是主要开销。

两个量谁更大取决于 **``M0`` 与 ``d/2`` 的比较**：
每个节点的边开销约 ``8 × M0`` 字节，向量开销是 ``4 × d`` 字节，
所以 ``M0 > d/2`` 时边更占。默认 M0=32、d=32 → 两者几乎持平；
如果 d=128（常见的 embedding 维度），向量反而重新成为大头。

先前这里写的是"几乎全部是边"——那是在 d=8、M=16 的小数据集上
得出的结论，换了维度就不成立。把它写成一句无条件的话，
会让读者在真实维度上做容量规划时算错一倍。

基准结果里会把两个数都列出来，不含糊。
"""

from __future__ import annotations

import heapq
import math
import random
from array import array
from dataclasses import dataclass, field

from .metrics import Metric, StaleIndexError, VectorStore, top_k


@dataclass
class HNSWConfig:
    """构建与查询参数。

    这些参数的含义必须分清，因为它们的**影响范围完全不同**：

    * ``m`` / ``ef_construction`` —— 影响**图的质量**，只在构建时用到。
      改它们必须重建索引。
    * ``ef_search`` —— 只影响**查询**，可以在查询时临时指定，
      不需要重建。这是 HNSW 最有用的性质：召回率可以按请求调。
    """

    #: 每层每个节点的邻居上限（第 0 层是 2 倍，见 ``m0``）。
    #: 论文建议 5~48。太小图不连通，太大内存暴涨且构建变慢。
    m: int = 16
    #: 构建时的候选池大小。越大图越好、构建越慢。
    #: 论文建议 100~500。
    ef_construction: int = 100
    #: 查询时的候选池大小。**这是召回率与延迟的直接旋钮**：
    #: 调大 → 召回率上升、延迟上升。可以查询时临时覆盖。
    ef_search: int = 64
    #: 层数采样用的种子。固定它是为了让**构建结果可复现** ——
    #: 否则同一个数据集两次构建出来的图不同，召回率会有一点抖动，
    #: 而"改了参数之后召回率涨了 1%"这种结论就完全不可信。
    seed: int = 20260930

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ValueError(f"m 必须 ≥ 2，收到 {self.m}")
        if self.ef_construction < self.m:
            # ef_construction < m 时选不满邻居，图会稀疏到不可用。
            # 与其让它跑出一个"看起来成功了但召回率只有 20%"的索引，
            # 不如在入口就拒绝。
            raise ValueError(
                f"ef_construction({self.ef_construction}) 不能小于 m({self.m})"
            )
        if self.ef_search < 1:
            raise ValueError(f"ef_search 必须 ≥ 1，收到 {self.ef_search}")

    @property
    def m0(self) -> int:
        """第 0 层的邻居上限。

        取 2M 是论文的推荐值。第 0 层要承担"精确落点"的职责，
        边越密越好；而上层只需要找对方向，2M 纯属浪费内存。
        """
        return self.m * 2

    @property
    def level_multiplier(self) -> float:
        """几何分布的尺度 ``mL = 1 / ln(M)``。

        这个取值让"层数超过 l 的节点比例"约为 ``exp(-l/mL) = M^(-l)``。
        代入 M=16：每上升一层，节点数变成约 1/16 —— 正好抵消掉
        每层贪心搜索的 O(M) 步，使总复杂度落在 O(log n)。
        用 M=16 而不是别的值，就是为了让这个抵消关系成立。
        """
        return 1.0 / math.log(self.m)

    def max_links(self, layer: int) -> int:
        return self.m0 if layer == 0 else self.m


@dataclass
class HNSWStats:
    nodes: int = 0
    levels: int = 0
    #: 所有层的边数之和。注意这不是任何"平均度"的正确分子 ——
    #: 平均度必须限定在同一层内算，见 ``layer0_edges``。
    edges: int = 0
    #: **仅第 0 层**的边数之和。第 0 层要承担"精确落点"的职责，
    #: 它的平均度是判断图是否够密的核心指标。
    layer0_edges: int = 0
    #: 各层的节点数，第 0 层在最前。用来直观看出"几何式衰减"是否成立。
    nodes_per_level: list[int] = field(default_factory=list)
    distance_computations: int = 0
    #: 服务过的查询次数。有了它才能算"平均每次查询算了多少次距离" ——
    #: 这是解释延迟最直接的一个量，也是唯一能把
    #: "图不够好"和" ef 设太大"区分开的证据。
    queries: int = 0
    #: **构建期**消耗的距离计算次数。
    #:
    #: 为什么要单独存一份：``distance_computations`` 是单调累加的，
    #: 它同时含了构建和查询。拿总量除以查询次数会得到
    #: "HNSW 每次查询比暴力检索还算得多"这种荒谬结论 ——
    #: 构建一次要算几十万次，摊到 25 条查询上就是每条约两万次，
    #: 而那几十万次跟任何一次查询都没关系。
    #: 拆开之后两个数字才有意义：构建成本是一次性的，
    #: 查询成本才是随请求线性增长的。
    construction_computations: int = 0
    build_seconds: float = 0.0

    @property
    def query_computations(self) -> int:
        """查询期消耗的距离计算次数（总量减去构建期的那部分）。"""
        return self.distance_computations - self.construction_computations


class HNSWIndex:
    """分层图索引。构建后只读。"""

    def __init__(
        self,
        store: VectorStore,
        metric: Metric,
        config: HNSWConfig | None = None,
    ) -> None:
        self.store = store
        self.metric = metric
        self.config = config or HNSWConfig()

        self._rng = random.Random(self.config.seed)
        #: ``_levels[key]`` = 该节点的最高层号
        self._levels: list[int] = []
        #: ``_links[key][layer]`` = 该层上的邻居 key 列表
        self._links: list[list[list[int]]] = []
        self._entry_point: int | None = None
        self._max_level: int = -1
        #: 构建时的向量库版本。查询时比对，防止在过期索引上查。
        self._built_revision: int | None = None
        self.stats = HNSWStats()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _distance_to_key(self, query: array, query_norm: float, key: int) -> float:
        item = self.store.get(key)
        self.stats.distance_computations += 1
        return self.metric.distance(query, query_norm, item.values, item.norm_sq)

    def _distance_between(self, a: int, b: int) -> float:
        """两个**库里**的向量之间的距离。

        用在邻居选择启发式里：判断"候选 c 是否离已选邻居 r 比离查询点更近"。
        注意查询点通常不在库里，所以这里不能用 ``_distance_to_key``。
        """
        first = self.store.get(a)
        second = self.store.get(b)
        self.stats.distance_computations += 1
        return self.metric.distance(
            first.values, first.norm_sq, second.values, second.norm_sq
        )

    def _sample_level(self) -> int:
        """按几何分布采样层号：``floor(-ln(U) × mL)``。

        为什么不是"每层以 1/M 概率继续往上"（那样更直观）：
        二者的分布等价，但这个写法少一次循环，而且**结果只依赖一个随机数**
        —— 少抽一个数就少一处可能影响可复现性的地方。
        """
        uniform = self._rng.random()
        # random() 可能返回 0.0（极小概率），log(0) 会抛 ValueError。
        # 用一个下界兜住，而不是 try/except —— 后者在热路径里更贵。
        uniform = max(uniform, 1e-12)
        return int(-math.log(uniform) * self.config.level_multiplier)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def build(self) -> "HNSWIndex":
        """把向量库里已有的全部向量建成图。"""
        self._levels = []
        self._links = []
        self._entry_point = None
        self._max_level = -1
        self.stats = HNSWStats()

        for item in self.store.items():
            self._insert(item.key)

        self._built_revision = self.store.revision
        self.stats.nodes = len(self._levels)
        self.stats.levels = self._max_level + 1
        self.stats.edges = sum(
            len(neighbours)
            for links in self._links
            for neighbours in links
        )
        # 第 0 层的边数必须**单独数**。曾经这里直接用上面那个总和
        # 去除以节点数来算"第 0 层平均度" ——
        # 分子含了所有层、分母只有第 0 层，得到的是一个
        # 没有物理意义的数字，而且永远偏大。
        # 症状很隐蔽：它看起来是个正常的度数值（比如 14），
        # 只有真的去数第 0 层的边才会发现对不上。
        self.stats.layer0_edges = sum(
            len(self._links[key][0]) for key in range(len(self._levels))
        )
        # 定格构建期的距离计算开销
        self.stats.construction_computations = self.stats.distance_computations
        self.stats.nodes_per_level = [
            sum(1 for lv in self._levels if lv >= layer)
            for layer in range(self._max_level + 1)
        ]
        return self

    def _insert(self, key: int) -> None:
        item = self.store.get(key)
        query = item.values
        query_norm = item.norm_sq

        level = self._sample_level()
        # 预留到 level 层（每层一个空邻居表）
        self._levels.append(level)
        self._links.append([[] for _ in range(level + 1)])

        if self._entry_point is None:
            # 第一个节点：它就是入口，没有任何边。
            self._entry_point = key
            self._max_level = level
            return

        entry = self._entry_point
        entry_distance = self._distance_to_key(query, query_norm, entry)

        # --- 阶段 1：从最高层贪心下降到 level+1 层 -------------------
        # 这一段只做 ef=1 的贪心。目的是**快速粗定位**：
        # 上层的边跨度大，几次跳跃就能从入口走到目标区域附近。
        # 这里不需要结果集，因为上层的目的只是"找一个好的下层入口"。
        for layer in range(self._max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbour in self._links[entry][layer]:
                    distance = self._distance_to_key(query, query_norm, neighbour)
                    if distance < entry_distance:
                        entry = neighbour
                        entry_distance = distance
                        changed = True

        # --- 阶段 2：从 min(level, max_level) 层到第 0 层，逐层插边 ---
        entry_points = [(entry_distance, entry)]
        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                query, query_norm, entry_points, self.config.ef_construction, layer
            )
            neighbours = self._select_neighbours(candidates, self.config.m)
            self._links[key][layer] = [k for _, k in neighbours]

            # 双向连边：新节点指过去，旧节点也要指回来。
            # 单向图在查询时可能"进得去出不来"，贪心会卡死在叶子上。
            for distance, other in neighbours:
                self._link_back(other, key, layer)

            # 下一层的入口用本层的搜索结果（而不是只取最近的那个）。
            # 给下层多个入口能让贪心从多个方向试，绕过局部最优。
            entry_points = candidates

        if level > self._max_level:
            # 新节点比当前最高层还高，它成为新的入口。
            self._entry_point = key
            self._max_level = level

    def _link_back(self, other: int, key: int, layer: int) -> None:
        """给 ``other`` 加上到 ``key`` 的边，超限就剪枝。"""
        neighbours = self._links[other][layer]
        if key in neighbours:
            return
        neighbours.append(key)

        limit = self.config.max_links(layer)
        if len(neighbours) <= limit:
            return

        # 超限：用同一个启发式重新挑。
        # **不能用"删掉最远的那个"** —— 那会系统性保留同一方向的边，
        # 把插入时好不容易维持的多样性重新破坏掉（见模块文档第 2 条）。
        scored = sorted((self._distance_between(other, n), n) for n in neighbours)
        kept = self._select_neighbours(scored, limit)
        self._links[other][layer] = [k for _, k in kept]

    def _search_layer(
        self,
        query: array,
        query_norm: float,
        entry_points: list[tuple[float, int]],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """在指定层上做带候选池的贪心搜索，返回最多 ``ef`` 个候选（距离升序）。

        这就是 HNSW 的核心循环，构建与查询共用同一份实现 ——
        共用是刻意的：构建与查询的搜索行为一旦不一致，
        召回率会取决于"插入顺序"这种看起来无关的东西，非常难查。
        """
        if not entry_points:
            return []

        # 去重：多个入口可能指向同一个节点（下层入口列表来自上层的搜索结果，
        # 本身可能有重复）。不去重会重复计算距离，而且结果里出现重复 key。
        seen: set[int] = set()
        frontier: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []  # 用负距离做最大堆

        for distance, key in entry_points:
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(frontier, (distance, key))
            heapq.heappush(best, (-distance, key))

        while len(best) > ef:
            heapq.heappop(best)

        while frontier:
            distance, current = heapq.heappop(frontier)
            # 早停：最近的可扩展候选已经比结果集里最差的还远，
            # 再往外走只会更远（贪心的单调性）。没有这一条，
            # 搜索会把整个连通分量走完 —— 结果相同，慢一个数量级。
            if best and distance > -best[0][0]:
                break

            for neighbour in self._links[current][layer]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                candidate = self._distance_to_key(query, query_norm, neighbour)
                if len(best) < ef or candidate < -best[0][0]:
                    heapq.heappush(frontier, (candidate, neighbour))
                    heapq.heappush(best, (-candidate, neighbour))
                    if len(best) > ef:
                        heapq.heappop(best)

        # 结果按距离升序。平局按 key 升序 —— 让输出**可复现**，
        # 否则同一份数据两次查询可能返回不同的 top-k 顺序。
        return sorted((-neg, key) for neg, key in best)

    def _select_neighbours(
        self, candidates: list[tuple[float, int]], limit: int
    ) -> list[tuple[float, int]]:
        """多样性启发式选邻居（论文 Algorithm 4）。

        规则：按距离从近到远考察候选 c，只有当 **c 到查询点的距离**
        小于 **c 到任何一个已选邻居的距离** 时才保留。

        直觉：如果 c 已经和某个已选邻居 r 很近了，那 c 能提供的"新方向"
        和 r 差不多，留它就是浪费一个邻居名额 —— 而这些名额在第 0 层
        是稀缺资源（只有 2M 个）。

        论文里的 ``extendCandidates`` / ``keepPrunedConnections`` 两个开关
        默认关闭，这里保持一致：多出来的候选带来的收益不明显，
        而多一遍距离计算在纯 Python 里是实打实的开销。
        """
        if limit <= 0:
            return []

        selected: list[tuple[float, int]] = []
        # 候选先按距离升序。这一步不能省：启发式必须"从近到远"考察，
        # 否则先选到的代表质量差，后面的判断全部基于坏代表。
        for distance, key in sorted(candidates):
            if len(selected) >= limit:
                break
            keep = True
            for _, chosen in selected:
                if self._distance_between(key, chosen) < distance:
                    keep = False
                    break
            if keep:
                selected.append((distance, key))
        return selected

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _ensure_fresh(self) -> None:
        if self._built_revision is None:
            raise RuntimeError("索引还没有构建，先调用 build()")
        if self.store.revision != self._built_revision:
            raise StaleIndexError(
                "向量库在索引构建之后被改过（revision "
                f"{self._built_revision} → {self.store.revision}）；"
                "请重建索引。继续在这个索引上查询会得到越来越差的结果，"
                "而且不会有任何报错。"
            )

    def search(
        self,
        query: array,
        query_norm: float,
        *,
        k: int = 10,
        ef_search: int | None = None,
    ) -> list[tuple[float, int]]:
        """返回 ``[(距离, key)]``，距离升序。

        ``ef_search`` 可以按查询临时覆盖 —— 这是 HNSW 最有用的性质：
        **召回率是查询期的旋钮**。同一个索引，低延迟场景用 ef=32，
        精度优先的场景用 ef=256，不需要重建。
        """
        self._ensure_fresh()
        self.stats.queries += 1
        if k <= 0:
            return []
        if self._entry_point is None:
            return []

        ef = max(ef_search or self.config.ef_search, k)
        # ef 必须 ≥ k：ef < k 时结果集本身就不够 k 个，
        # 返回的结果数量会不足，而调用方通常不会检查这一点 ——
        # 它会以为"只找到 5 个"是因为库里就这么少。
        # 夹取到 k 保证"要么给够 k 个，要么是真的没那么多"。

        entry = self._entry_point
        entry_distance = self._distance_to_key(query, query_norm, entry)
        current = entry
        current_distance = entry_distance

        # 自顶向下，每层只做 ef=1 的贪心
        for layer in range(self._max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbour in self._links[current][layer]:
                    distance = self._distance_to_key(query, query_norm, neighbour)
                    if distance < current_distance:
                        current = neighbour
                        current_distance = distance
                        changed = True

        candidates = self._search_layer(
            query, query_norm, [(current_distance, current)], ef, 0
        )
        return top_k(((distance, key) for distance, key in candidates), k)

    # ------------------------------------------------------------------
    # 核算
    # ------------------------------------------------------------------

    def memory_bytes(self) -> dict[str, int]:
        """内存构成。

        分成三块分别报，因为它们的优化方向完全不同：

        * ``vectors``     —— 向量本体。要压小只能量化（PQ 干的事）；
        * ``links``       —— 边。要压小只能降 M 或降层数；
        * ``level_table`` —— 每节点一个 int 的层号，可以忽略不计。

        ``links`` 按"每条边一个 8 字节槽位 + 一个指向 int 的引用"算。
        这里报的是**逻辑上界**（每个边位 8 字节），不报 CPython 里
        list 的过度分配与实际对象头 —— 界面上会把这个口径写清楚，
        免得读者把它当成 RSS。
        """
        link_slots = sum(len(neighbours) for links in self._links for neighbours in links)
        return {
            "vectors": self.store.memory_bytes(),
            "links": link_slots * 8,
            "link_slots": link_slots,
            "level_table": len(self._levels) * 8,
            "total": self.store.memory_bytes() + link_slots * 8 + len(self._levels) * 8,
        }

    def snapshot(self) -> dict[str, object]:
        mem = self.memory_bytes()
        return {
            "nodes": self.stats.nodes,
            "levels": self.stats.levels,
            "edges": self.stats.edges,
            "layer0_edges": self.stats.layer0_edges,
            "nodes_per_level": self.stats.nodes_per_level,
            "construction_computations": self.stats.construction_computations,
            # 还没服务过查询时是 0 而不是抛 ZeroDivisionError ——
            # snapshot() 要能在"刚建好、一条查询都还没来"的时候被调用，
            # 那正是前端展示"构建结果"的第一步。
            "avg_distance_per_query": (
                round(
                    self.stats.query_computations / self.stats.queries, 1
                )
                if self.stats.queries
                else 0.0
            ),
            "avg_degree_layer0": round(
                self.stats.layer0_edges / max(self.stats.nodes, 1), 3
            ),
            "memory": mem,
            "config": {
                "m": self.config.m,
                "m0": self.config.m0,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "seed": self.config.seed,
                "level_multiplier": round(self.config.level_multiplier, 4),
            },
            "build_seconds": round(self.stats.build_seconds, 4),
            "distance_computations": self.stats.distance_computations,
        }
