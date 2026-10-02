"""暴力检索与召回率口径的测试。

暴力检索是**所有召回率数字的裁判**，它若不可靠，整个 benchmark 就是空的。
召回率的分母口径也在这里定死 —— 这部分不是学术洁癖，
分母选错会让"IVF-PQ 比 HNSW 差多少"这种结论整个翻转。
"""

from __future__ import annotations

import pytest

from app.bruteforce import (
    ExactResult,
    agreement,
    exact_top_k,
    recall_at_k,
    threshold_recall_at_k,
)
from app.metrics import VectorStore, make_metric, squared_norm
from tests.helpers import add_all, as_array, grid_vectors, make_store


def _store_with(dim: int, vectors, metric: str = "l2"):
    store = VectorStore(dim, make_metric(metric))
    add_all(store, vectors)
    return store


class TestExactTopK:
    def test_finds_nearest_neighbour(self):
        dim = 3
        store = _store_with(dim, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        query = store.prepare_query([1.1, 0.0, 0.0])
        result = exact_top_k(store, query, make_metric("l2"), k=1)
        assert result.keys == [1]

    def test_results_are_sorted_by_distance(self):
        dim = 3
        store = _store_with(dim, [[4.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [0.0, 0, 0]])
        query = store.prepare_query([0.0, 0.0, 0.0])
        result = exact_top_k(store, query, make_metric("l2"), k=4)
        assert result.keys == [3, 1, 2, 0]
        dists = result.distances
        assert dists == sorted(dists)

    def test_query_at_zero_gives_zero_distance_to_itself(self):
        store = make_store(dim=4)
        add_all(store, [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        query = store.prepare_query([0.0, 0.0, 0.0, 0.0])
        result = exact_top_k(store, query, make_metric("l2"), k=1)
        assert result.distances[0] == pytest.approx(0.0)

    def test_k_larger_than_corpus_returns_all(self):
        store = make_store(dim=4)
        add_all(store, [[1.0, 0, 0, 0], [2.0, 0, 0, 0]])
        query = store.prepare_query([1.5, 0.0, 0.0, 0.0])
        result = exact_top_k(store, query, make_metric("l2"), k=10)
        assert len(result.hits) == 2

    def test_k_zero_rejected(self):
        store = make_store(dim=4)
        store.add([1.0, 0, 0, 0])
        query = store.prepare_query([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="k 必须为正整数"):
            exact_top_k(store, query, make_metric("l2"), k=0)

    def test_negative_k_rejected(self):
        store = make_store(dim=4)
        store.add([1.0, 0, 0, 0])
        query = store.prepare_query([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="k 必须为正整数"):
            exact_top_k(store, query, make_metric("l2"), k=-3)

    def test_empty_store_gives_empty_result(self):
        store = make_store(dim=4)
        query = store.prepare_query([1.0, 0.0, 0.0, 0.0])
        result = exact_top_k(store, query, make_metric("l2"), k=5)
        assert result.hits == []

    def test_result_records_metric_and_k(self):
        store = make_store(dim=4)
        store.add([1.0, 0, 0, 0])
        query = store.prepare_query([1.0, 0.0, 0.0, 0.0])
        result = exact_top_k(store, query, make_metric("cosine"), k=2)
        assert result.metric == "cosine"
        assert result.k == 2

    def test_deterministic_across_calls(self):
        store = _store_with(3, [[1.0, 0, 0], [1.0, 0, 0], [0.9, 0, 0]])
        query = store.prepare_query([1.0, 0.0, 0.0])
        a = exact_top_k(store, query, make_metric("l2"), k=2)
        b = exact_top_k(store, query, make_metric("l2"), k=2)
        assert agreement(a, b) == 1.0
        assert a.keys == b.keys

    def test_grid_corpus_places_nearest_first(self):
        """网格数据集上，最近邻一定是查询点自己所在的那个格子。"""
        store = make_store(dim=3)
        grid = grid_vectors(store, side=3)
        add_all(store, grid)
        query = store.prepare_query(grid[13])
        result = exact_top_k(store, query, make_metric("l2"), k=1)
        assert result.keys == [13]


class TestExactResultThreshold:
    def test_threshold_is_kth_distance(self):
        hits = [(1.0, 0), (2.0, 1), (3.0, 2)]
        result = ExactResult(hits=hits, metric="l2", k=3)
        assert result.threshold == pytest.approx(3.0)

    def test_threshold_is_inf_when_fewer_than_k_hits(self):
        """不足 k 个结果时必须返回 inf 而不是 0。

        返回 0 会让所有距离大于 0 的 ANN 结果都被判成"未命中"，
        召回率恒等于 0 —— 一个静默的错误数字，比崩溃更难发现。
        """
        hits = [(1.0, 0), (2.0, 1)]
        result = ExactResult(hits=hits, metric="l2", k=10)
        assert result.threshold == float("inf")

    def test_threshold_is_inf_when_no_hits(self):
        assert ExactResult(hits=[], metric="l2", k=5).threshold == float("inf")

    def test_threshold_finite_when_exactly_k_hits(self):
        hits = [(1.0, 0), (2.0, 1)]
        assert ExactResult(hits=hits, metric="l2", k=2).threshold == pytest.approx(2.0)

    def test_to_wire_limits_and_rounds(self):
        hits = [(1.23456789, 0), (2.0, 1), (3.0, 2)]
        wire = ExactResult(hits=hits, metric="l2", k=3).to_wire(limit=2)
        assert len(wire) == 2
        assert wire[0] == {"distance": 1.234568, "key": 0}


class TestRecallAtK:
    def _exact(self, keys, k=3):
        return ExactResult(hits=[(float(i), key) for i, key in enumerate(keys)],
                           metric="l2", k=k)

    def test_perfect_overlap(self):
        exact = self._exact([1, 2, 3])
        assert recall_at_k([1, 2, 3], exact) == 1.0

    def test_partial_overlap(self):
        exact = self._exact([1, 2, 3])
        assert recall_at_k([1, 2, 99], exact) == pytest.approx(2 / 3)

    def test_no_overlap(self):
        exact = self._exact([1, 2, 3])
        assert recall_at_k([7, 8, 9], exact) == 0.0

    def test_denominator_is_k_not_number_of_exact_hits(self):
        """库里只有 2 个点而 k=10 时，召回率上限就是 2/10。

        用 ``len(exact.keys)`` 做分母会得到 2/2 = 1.0，
        于是"库太小"和"索引很强"在数据上完全无法区分。
        """
        exact = self._exact([1, 2], k=10)
        assert recall_at_k([1, 2], exact) == pytest.approx(0.2)

    def test_zero_k_gives_zero(self):
        exact = ExactResult(hits=[(1.0, 0)], metric="l2", k=0)
        assert recall_at_k([0], exact) == 0.0

    def test_duplicate_keys_in_ann_do_not_inflate(self):
        """ANN 返回重复 key 不能重复计数。"""
        exact = self._exact([1, 2, 3])
        assert recall_at_k([1, 1, 1], exact) == pytest.approx(1 / 3)


class TestThresholdRecallAtK:
    def _exact(self, keys, distances, k):
        return ExactResult(
            hits=list(zip(distances, keys)), metric="l2", k=k
        )

    def test_all_within_threshold(self):
        exact = self._exact([0, 1, 2], [1.0, 2.0, 3.0], 3)
        ann = [(0.5, 0), (1.5, 1), (2.5, 2)]
        assert threshold_recall_at_k(ann, exact) == 1.0

    def test_partial_within_threshold(self):
        exact = self._exact([0, 1, 2], [1.0, 2.0, 3.0], 3)
        ann = [(0.5, 0), (1.5, 1), (99.0, 7)]
        assert threshold_recall_at_k(ann, exact) == pytest.approx(2 / 3)

    def test_exactly_at_threshold_counts(self):
        exact = self._exact([0, 1, 2], [1.0, 2.0, 3.0], 3)
        ann = [(3.0, 0), (3.0, 1), (3.0, 2)]
        assert threshold_recall_at_k(ann, exact) == 1.0

    def test_float_noise_at_threshold_still_counts(self):
        """同一份数据算出 1e-15 的差，不能判为未命中。

        PQ 的近似距离和精确距离经常差这么一点；
        不加容差会让"并列边缘"的命中随机翻脸。
        """
        exact = self._exact([0, 1, 2], [1.0, 2.0, 3.0], 3)
        ann = [(3.0 + 1e-12, 0)]
        assert threshold_recall_at_k(ann, exact) >= 1 / 3

    def test_inf_threshold_means_everything_counts(self):
        exact = self._exact([0, 1], [1.0, 2.0], 10)
        ann = [(500.0, 0), (600.0, 1)]
        assert threshold_recall_at_k(ann, exact) == pytest.approx(2 / 10)

    def test_never_exceeds_one(self):
        """ANN 返回超过 k 个结果时不能让召回率大于 1。"""
        exact = self._exact([0, 1, 2], [1.0, 2.0, 3.0], 3)
        ann = [(0.1, i) for i in range(20)]
        assert threshold_recall_at_k(ann, exact) == 1.0

    def test_zero_k_gives_zero(self):
        exact = ExactResult(hits=[(1.0, 0)], metric="l2", k=0)
        assert threshold_recall_at_k([(1.0, 0)], exact) == 0.0


class TestRecallDefinitionsDiverge:
    """两种召回率在有并列时会给出不同甚至相反的答案。

    这组测试要钉住的是"它们确实会分岔"这件事本身。
    benchmark 里同时给出两个数字，就是为了让读者看到这个分岔，
    而不是拿一个单次数字去下结论。
    """

    def test_standard_recall_punishes_tie_breaking(self):
        """并列时 ANN 选了不同的 key，标准召回率会判它失分。"""
        # 精确 top-3 = key 0/1/2，但 2/3/4 与它们距离完全相同
        exact = ExactResult(
            hits=[(1.0, 0), (1.0, 1), (1.0, 2)], metric="l2", k=3
        )
        # ANN 挑了等距的另一组
        ann_keys = [2, 3, 4]
        std = recall_at_k(ann_keys, exact)
        thr = threshold_recall_at_k([(1.0, 2), (1.0, 3), (1.0, 4)], exact)
        assert std < thr
        assert thr == 1.0

    def test_both_agree_without_ties(self):
        exact = ExactResult(
            hits=[(1.0, 0), (2.0, 1), (3.0, 2)], metric="l2", k=3
        )
        ann = [(1.0, 0), (2.0, 1), (3.0, 2)]
        assert recall_at_k([0, 1, 2], exact) == 1.0
        assert threshold_recall_at_k(ann, exact) == 1.0


class TestAgreement:
    def test_identical_results_agree_fully(self):
        hits = [(1.0, 0), (2.0, 1)]
        a = ExactResult(hits=list(hits), metric="l2", k=2)
        b = ExactResult(hits=list(hits), metric="l2", k=2)
        assert agreement(a, b) == 1.0

    def test_disjoint_results_agree_nothing(self):
        a = ExactResult(hits=[(1.0, 0)], metric="l2", k=1)
        b = ExactResult(hits=[(1.0, 1)], metric="l2", k=1)
        assert agreement(a, b) == 0.0

    def test_both_empty_is_full_agreement(self):
        a = ExactResult(hits=[], metric="l2", k=1)
        b = ExactResult(hits=[], metric="l2", k=1)
        assert agreement(a, b) == 1.0

    def test_one_empty_is_zero_agreement(self):
        a = ExactResult(hits=[], metric="l2", k=1)
        b = ExactResult(hits=[(1.0, 0)], metric="l2", k=1)
        assert agreement(a, b) == 0.0


class TestStoreHelperIntegrity:
    def test_squared_norm_matches_store_normalization(self):
        store = make_store(dim=8)
        store.add([3.0, 4.0] + [0.0] * 6)
        values = store.values_of(0)
        assert squared_norm(values) == pytest.approx(25.0)

    def test_as_array_helper(self):
        assert list(as_array([1.0, 2.0])) == [1.0, 2.0]
