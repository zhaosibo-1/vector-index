"""HNSW 图索引的测试。

不测"召回率能到 1.0"这种碰运气的事，测的是**结构性质**：
层数是不是按几何律衰减、边是不是双向的、是否 Hermitian、
多样性启发式有没有真的在起作用、ef 是不是单调的旋钮。
这些性质在 HNSW 论文里往往被一笔带过，但每一项写错都会让
索引在大数据集上安静地退化。
"""

from __future__ import annotations

import pytest

from app.bruteforce import exact_top_k, recall_at_k
from app.hnsw import HNSWConfig, HNSWIndex, StaleIndexError
from app.metrics import VectorStore, make_metric
from tests.helpers import grid_vectors, make_store


def build_index(n=160, dim=8, metric="l2", data_seed=1, **cfg):
    """搭一个小数据集上的索引。

    ``**cfg`` 原样交给 ``HNSWConfig``（m、ef_search、seed 等）。
    数据集的 ``data_seed`` 与索引的 ``seed`` 是**两个不同的东西**：
    前者决定库里有哪些点，后者决定层号怎么采样。
    混成一个参数会让"换种子"这个实验同时改两处，
    得到的差异说不清是哪一半造成的。
    """
    import random

    rng = random.Random(data_seed)
    metric_obj = make_metric(metric)
    store = VectorStore(dim, metric_obj)
    for _ in range(n):
        store.add([rng.gauss(0.0, 1.0) for _ in range(dim)])
    index = HNSWIndex(store, metric_obj, HNSWConfig(**cfg))
    index.build()
    return store, metric_obj, index


class TestConfig:
    def test_m_too_small_rejected(self):
        with pytest.raises(ValueError, match="m"):
            HNSWConfig(m=1)

    def test_ef_construction_below_m_rejected(self):
        """ef_construction < m 时选不满邻居，图会稀疏到不可用。

        与其让它跑出一个"看起来成功了但召回率只有 20%"的索引，
        不如在入口拒绝。
        """
        with pytest.raises(ValueError, match="不能小于"):
            HNSWConfig(m=16, ef_construction=8)

    def test_ef_search_zero_rejected(self):
        with pytest.raises(ValueError, match="ef_search"):
            HNSWConfig(ef_search=0)

    def test_m0_is_double_m(self):
        assert HNSWConfig(m=16).m0 == 32

    def test_level_multiplier_is_inverse_log_m(self):
        import math

        assert HNSWConfig(m=16).level_multiplier == pytest.approx(
            1.0 / math.log(16)
        )

    def test_max_links_differs_per_layer(self):
        cfg = HNSWConfig(m=16)
        assert cfg.max_links(0) == 32
        assert cfg.max_links(1) == 16


class TestBuildStructure:
    def test_all_nodes_present(self):
        store, _, index = build_index(n=120)
        assert index.stats.nodes == 120
        assert len(index._levels) == 120

    def test_layer_zero_holds_everybody(self):
        store, _, index = build_index(n=120)
        assert index.stats.nodes_per_level[0] == 120

    def test_higher_layers_shrink_geometrically(self):
        """每上升一层节点数约为 1/M —— 这是 O(log n) 的来源。

        用的是 M=6（而不是默认 16），小样本才看得出层次。
        """
        store, _, index = build_index(n=800, m=6)
        levels = index.stats.nodes_per_level
        assert len(levels) >= 2
        for i in range(1, len(levels)):
            assert levels[i] < levels[i - 1]

    def test_entry_point_is_on_top_layer(self):
        store, _, index = build_index(n=120)
        assert index._max_level == index._levels[index._entry_point]

    def test_edges_are_bidirectional(self):
        """双向边是 HNSW 泄漏最少的一个细节。

        单向边会让"从某点出发可达"和"能到达某点"不对称，
        搜索会沿着单向边走过去却回不来，结果随起点不同而不同。
        """
        store, _, index = build_index(n=200, seed=3)
        bad = 0
        for key in range(len(store)):
            for layer in range(index._levels[key] + 1):
                for other in index._links[key][layer]:
                    if key not in index._links[other][layer]:
                        bad += 1
        assert bad == 0

    def test_no_self_loops(self):
        store, _, index = build_index(n=120, seed=4)
        for key in range(len(store)):
            for layer in range(index._levels[key] + 1):
                assert key not in index._links[key][layer]

    def test_no_duplicate_neighbours(self):
        store, _, index = build_index(n=120, seed=5)
        for key in range(len(store)):
            for layer in range(index._levels[key] + 1):
                links = index._links[key][layer]
                assert len(links) == len(set(links))

    def test_neighbour_cap_respected(self):
        store, _, index = build_index(n=200, seed=6, m=8)
        cfg = index.config
        for key in range(len(store)):
            for layer in range(index._levels[key] + 1):
                assert len(index._links[key][layer]) <= cfg.max_links(layer)

    def test_build_is_deterministic(self):
        """固定 seed 必须给出同样的图。"""
        a = build_index(n=140, data_seed=9)[2]
        b = build_index(n=140, data_seed=9)[2]
        assert a._levels == b._levels
        assert a._links == b._links

    def test_different_seed_gives_different_graph(self):
        a = build_index(n=140, data_seed=9)[2]
        b = build_index(n=140, data_seed=11)[2]
        assert a._levels != b._levels or a._links != b._links


class TestSearch:
    def test_returns_k_results(self):
        store, metric, index = build_index(n=120, seed=2)
        q = store.prepare_query([0.5] * 8)
        assert len(index.search(q[0], q[1], k=10)) == 10

    def test_results_sorted_ascending(self):
        store, metric, index = build_index(n=120, seed=2)
        q = store.prepare_query([0.5] * 8)
        hits = index.search(q[0], q[1], k=10)
        assert [d for d, _ in hits] == sorted(d for d, _ in hits)

    def test_keys_are_all_distinct(self):
        store, metric, index = build_index(n=120, seed=2)
        q = store.prepare_query([0.5] * 8)
        hits = index.search(q[0], q[1], k=10)
        keys = [k for _, k in hits]
        assert len(keys) == len(set(keys))

    def test_matches_exact_nearest_neighbour(self):
        """top-1 必须与暴力检索一致。

        这条测的是"图有没有连通"：层之间哪怕有一条边没连上，
        某些区域的点就永远不可达，表现为偶然查错最近的那个。
        """
        store, metric, index = build_index(n=200, seed=7)
        for i in range(20):
            q = store.prepare_query(store.values_of(i))
            exact_nn = min(
                (d, k)
                for d, k in [
                    (metric.distance(q[0], q[1], store.values_of(key),
                                     store.get(key).norm_sq), key)
                    for key in store.all_keys()
                ]
            )
            ann_hits = index.search(q[0], q[1], k=1)
            assert ann_hits[0][1] == exact_nn[1]

    def test_high_recall_on_small_set(self):
        store, metric, index = build_index(n=200, seed=7)
        total = 0.0
        trials = 20
        for i in range(trials):
            q = store.prepare_query(store.values_of(i))
            exact = exact_top_k(store, q, metric, k=10)
            hits = index.search(q[0], q[1], k=10, ef_search=200)
            total += recall_at_k([k for _, k in hits], exact)
        assert total / trials >= 0.95

    def test_ef_search_is_a_monotone_knob(self):
        """ef 越大召回率不降 —— 这是它作为旋钮的前提。"""
        store, metric, index = build_index(n=300, seed=11, ef_search=8)
        small = _mean_recall(store, metric, index, ef=8)
        large = _mean_recall(store, metric, index, ef=128)
        assert large >= small

    def test_ef_search_is_clamped_to_at_least_k(self):
        """ef < k 时必须补到 k，否则结果数量悄悄不足。"""
        store, metric, index = build_index(n=120, seed=2, ef_search=1)
        q = store.prepare_query([0.5] * 8)
        hits = index.search(q[0], q[1], k=10, ef_search=1)
        assert len(hits) == 10

    def test_k_larger_than_corpus(self):
        store, metric, index = build_index(n=20, seed=2)
        q = store.prepare_query([0.5] * 8)
        assert len(index.search(q[0], q[1], k=50)) == 20

    def test_zero_k_returns_empty(self):
        store, metric, index = build_index(n=50, seed=2)
        q = store.prepare_query([0.5] * 8)
        assert index.search(q[0], q[1], k=0) == []

    def test_negative_k_returns_empty(self):
        """负 k 是调用方算错，返回空比抛错温和 —— 和 top_k 一致。"""
        store, metric, index = build_index(n=50, seed=2)
        q = store.prepare_query([0.5] * 8)
        assert index.search(q[0], q[1], k=-3) == []

    def test_query_is_idempotent(self):
        """同一个查询连查两次必须完全一样。

        索引内部若有隐藏状态（比如复用了某个候选列表），
        第二次查询的结果就会带上第一次的残留。
        """
        store, metric, index = build_index(n=120, seed=2, m=8)
        q = store.prepare_query([0.7] * 8)
        first = index.search(q[0], q[1], k=5)
        second = index.search(q[0], q[1], k=5)
        assert first == second


class TestStaleIndex:
    def test_adding_vector_after_build_raises(self):
        """索引不会因为库变了而报错 —— 它会安静返回越来越差的结果。

        这是整类"线上召回率随时间下降"事故的根因，
        所以这里必须硬失败。
        """
        store, metric, index = build_index(n=60, seed=2)
        store.add([1.0] * 8)
        q = store.prepare_query([0.5] * 8)
        with pytest.raises(StaleIndexError):
            index.search(q[0], q[1], k=5)

    def test_replacing_vector_after_build_raises(self):
        store, metric, index = build_index(n=60, seed=2)
        store.replace(0, [3.0] * 8)
        q = store.prepare_query([0.5] * 8)
        with pytest.raises(StaleIndexError):
            index.search(q[0], q[1], k=5)

    def test_rebuild_restores_usability(self):
        store, metric, index = build_index(n=60, seed=2)
        store.add([1.0] * 8)
        index.build()
        q = store.prepare_query([0.5] * 8)
        assert len(index.search(q[0], q[1], k=5)) == 5

    def test_search_before_build_raises_loudly(self):
        """还没 build 就 search，必须明确报错而不是返回空列表。

        返回空会让上层以为"库里没有匹配项"，
        真正的原因（忘了 build）则被彻底掩盖。
        """
        store = VectorStore(8, make_metric("l2"))
        for i in range(10):
            store.add([float(i)] * 8)
        index = HNSWIndex(store, make_metric("l2"), HNSWConfig())
        q = store.prepare_query([1.0] * 8)
        with pytest.raises(RuntimeError, match="还没有构建"):
            index.search(q[0], q[1], k=3)


class TestDiversityHeuristic:
    """多样性启发式（论文 Algorithm 4）的行为。

    判据是：**候选 c 到查询的距离 < c 到任一已选邻居的距离**时才保留。
    距离矩阵用一个假的 ``_distance_between`` 接管，几何关系完全可控 ——
    用真实数据集写这个测试会得到一个"依赖具体随机几何"的断言，
    换 seed 就红，而语义其实没变。

    注意判据是严格 ``<``（不是 ``<=``）：与某个已选邻居**恰好等距**的
    候选会被保留。这是论文原文的行为，不是笔误 ——
    等距意味着两者在不同的方向上，本来就不冗余。
    """

    @staticmethod
    def _fake_distance(matrix):
        def call(a, b):
            if a == b:
                return 0.0
            return matrix.get((a, b), matrix.get((b, a), 100.0))
        return call

    def test_drops_candidate_that_is_closer_to_a_selected_neighbour(self):
        store, metric, index = build_index(n=10, seed=13)
        # 2 与 0 相距 0.1，远小于 2 到查询的 3.0 → 2 是冗余的
        matrix = {(0, 1): 5.0, (0, 2): 0.1, (1, 2): 7.0}
        index._distance_between = self._fake_distance(matrix)
        cands = [(1.0, 0), (2.0, 1), (3.0, 2)]
        assert [k for _, k in index._select_neighbours(cands, 5)] == [0, 1]

    def test_keeps_candidate_that_points_elsewhere(self):
        store, metric, index = build_index(n=10, seed=13)
        # 2 到谁都远 → 它代表一个新方向，必须留下
        matrix = {(0, 1): 9.0, (0, 2): 9.0, (1, 2): 9.0}
        index._distance_between = self._fake_distance(matrix)
        cands = [(1.0, 0), (2.0, 1), (3.0, 2)]
        assert [k for _, k in index._select_neighbours(cands, 5)] == [0, 1, 2]

    def test_exactly_equidistant_candidate_is_kept(self):
        """与已选邻居恰好等距 → 保留（严格小于）。"""
        store, metric, index = build_index(n=10, seed=13)
        matrix = {(0, 1): 2.0}
        index._distance_between = self._fake_distance(matrix)
        cands = [(1.0, 0), (2.0, 1)]
        assert [k for _, k in index._select_neighbours(cands, 5)] == [0, 1]

    def test_select_neighbours_respects_cap(self):
        store, metric, index = build_index(n=10, seed=13)
        index._distance_between = self._fake_distance({})
        cands = [(float(i), i) for i in range(50)]
        assert len(index._select_neighbours(cands, 8)) == 8

    def test_empty_candidates_gives_empty(self):
        store, metric, index = build_index(n=10, seed=13)
        assert index._select_neighbours([], 8) == []

    def test_zero_limit_gives_empty(self):
        store, metric, index = build_index(n=10, seed=13)
        assert index._select_neighbours([(1.0, 0)], 0) == []

    def test_input_is_not_mutated(self):
        """候选列表不能被就地改动。

        调用方（插入路径）会把同一个列表交给别的分支，
        就地排序会让那一侧看到被重排过的数据。
        """
        store, metric, index = build_index(n=10, seed=13)
        index._distance_between = self._fake_distance({(0, 1): 9.0})
        cands = [(3.0, 2), (1.0, 0), (2.0, 1)]
        snapshot_order = list(cands)
        index._select_neighbours(cands, 5)
        assert cands == snapshot_order


class TestMemoryAccounting:
    def test_memory_breakdown_sums_to_total(self):
        store, metric, index = build_index(n=150, seed=2)
        mem = index.memory_bytes()
        assert mem["vectors"] + mem["links"] + mem["level_table"] == mem["total"]

    def test_edges_cost_more_than_vectors(self):
        """这是 HNSW 内存账的关键结论：图结构比向量本身更占。

        很多人以为"向量数据库内存 = 向量内存"，
        实测下来边表一项就能翻倍 —— 这也是为什么
        "能不能塞进内存"不能按 vector count 估算。
        """
        store, metric, index = build_index(n=400, seed=2, m=16)
        mem = index.memory_bytes()
        assert mem["links"] > mem["vectors"]

    def test_vectors_field_matches_store(self):
        store, metric, index = build_index(n=150, seed=2)
        assert index.memory_bytes()["vectors"] == store.memory_bytes()


class TestSnapshot:
    def test_snapshot_contract(self):
        store, metric, index = build_index(n=120, seed=2)
        snap = index.snapshot()
        for field_name in (
            "nodes", "levels", "edges", "nodes_per_level",
            "avg_degree_layer0", "memory", "config", "distance_computations",
        ):
            assert field_name in snap, field_name

    def test_avg_degree_matches_count(self):
        store, metric, index = build_index(n=150, seed=2)
        snap = index.snapshot()
        # 分子必须是**第 0 层**的边数，不是所有层之和
        assert snap["avg_degree_layer0"] == pytest.approx(
            snap["layer0_edges"] / snap["nodes_per_level"][0], abs=1e-3
        )
        assert snap["layer0_edges"] <= snap["edges"]

    def test_distance_computations_is_counted(self):
        store, metric, index = build_index(n=80, seed=2)
        index.search(*store.prepare_query([0.5] * 8), k=5)
        assert index.stats.distance_computations > 0


class TestCosineMetric:
    def test_cosine_index_search_works(self):
        store, metric, index = build_index(n=120, dim=8, metric="cosine", seed=2)
        q = store.prepare_query(store.values_of(0))
        hits = index.search(q[0], q[1], k=5)
        assert len(hits) == 5
        assert hits[0][1] == 0

    def test_cosine_distances_are_non_negative(self):
        store, metric, index = build_index(n=120, dim=8, metric="cosine", seed=2)
        q = store.prepare_query(store.values_of(0))
        hits = index.search(q[0], q[1], k=5)
        assert all(d >= -1e-9 for d, _ in hits)


def _mean_recall(store, metric, index, ef, trials=15):
    total = 0.0
    for i in range(trials):
        q = store.prepare_query(store.values_of(i))
        exact = exact_top_k(store, q, metric, k=10)
        hits = index.search(q[0], q[1], k=10, ef_search=ef)
        total += recall_at_k([k for _, k in hits], exact)
    return total / trials
