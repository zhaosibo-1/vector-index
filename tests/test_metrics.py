"""度量层的测试。

这一层是全项目正确性的地基：图索引和量化索引都在它的距离函数上跑，
它算错一格，上面两层会一起错、而且错得没有规律。

重点测三件容易被想当然糊弄过去的事：
1. 三种度量的距离方向统一为"越小越近"；
2. cosine 的归一化是在入库和查询**两侧**都做了；
3. 距离值**不**被 clamp 到 0（clamp 会毁掉排序的稳定性）。
"""

from __future__ import annotations

import math

import pytest

from app.metrics import (
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    StoredVector,
    VectorStore,
    dot,
    make_metric,
    squared_norm,
    top_k,
)
from tests.helpers import add_all, as_array, grid_vectors, make_store


class TestDotAndNorm:
    def test_dot_matches_manual_sum(self):
        assert dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)

    def test_squared_norm_is_square_of_euclidean(self):
        assert squared_norm([3, 4]) == pytest.approx(25.0)

    def test_sumprod_fallback_gives_same_answer(self):
        """原生 ``math.sumprod`` 与手写展开必须一致。

        降级路径不是"反正很少走到"的分支：它只在 Python < 3.12 上生效，
        但在那些版本上它承担全部计算。两条路结果不一样，
        意味着换 Python 版本会换检索结果。
        """
        a = [0.1 * i for i in range(16)]
        b = [math.pi * (i - 3) for i in range(16)]
        manual = sum(x * y for x, y in zip(a, b))
        assert dot(a, b) == pytest.approx(manual)


class TestMetricCreation:
    def test_valid_names(self):
        for name in (METRIC_L2, METRIC_COSINE, METRIC_INNER_PRODUCT):
            assert make_metric(name).name == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="未知的度量"):
            make_metric("manhattan")

    def test_cosine_needs_normalization(self):
        assert make_metric(METRIC_COSINE).needs_normalization is True
        assert make_metric(METRIC_L2).needs_normalization is False

    def test_inner_product_is_similarity_not_distance(self):
        assert make_metric(METRIC_INNER_PRODUCT).is_similarity is True
        assert make_metric(METRIC_L2).is_similarity is False


class TestL2Distance:
    def test_zero_for_identical_vectors(self):
        m = make_metric(METRIC_L2)
        v = as_array([1.0, 2.0, 3.0])
        norm = squared_norm(v)
        assert m.distance(v, norm, v, norm) == pytest.approx(0.0)

    def test_matches_squared_euclidean(self):
        m = make_metric(METRIC_L2)
        a = as_array([0.0, 0.0, 0.0])
        b = as_array([3.0, 4.0, 0.0])
        assert m.distance(a, 0.0, b, 25.0) == pytest.approx(25.0)

    def test_not_clamped_at_zero_for_near_identical(self):
        """近重复向量的距离必须严格为正。

        把负值 clamp 到 0 会让"几乎重合"和"完全重合"不可区分，
        top-k 的并列打平就变成随机的 —— 这是召回率抖动的经典来源。

        顺便钉住一个容易忘的事实：向量以 float32 存储，
        差异小于约 1e-7 时会被存储层直接吃掉，距离变成 0。
        这是**有意的取舍**（省一半内存），不是 bug；
        但如果哪天有人把它"修好"成 float64，这个测试会提醒他
        内存指标的面貌也会跟着变。
        """
        m = make_metric(METRIC_L2)
        a = as_array([1.0, 1.0])
        b = as_array([1.0 + 1e-4, 1.0])
        d = m.distance(a, 2.0, b, squared_norm(b))
        assert d == pytest.approx(1e-8, rel=1e-3)
        assert d > 0.0

    def test_float32_swallows_differences_below_epsilon(self):
        """太小 ✅ 的差异会被 float32 吃掉，距离恒为 0。"""
        m = make_metric(METRIC_L2)
        a = as_array([1.0, 1.0])
        b = as_array([1.0 + 1e-9, 1.0])
        assert m.distance(a, 2.0, b, squared_norm(b)) == pytest.approx(0.0)


class TestCosineDistance:
    def test_orthogonal_vectors_are_far(self):
        m = make_metric(METRIC_COSINE)
        a = as_array([1.0] + [0.0] * 7)
        b = as_array([0.0] * 7 + [5.0])
        assert m.distance(a, 1.0, b, 25.0) == pytest.approx(1.0)

    def test_parallel_vectors_are_close(self):
        m = make_metric(METRIC_COSINE)
        a = as_array([1.0] + [0.0] * 7)
        b = as_array([7.0] + [0.0] * 7)
        assert m.distance(a, 1.0, b, 49.0) == pytest.approx(0.0)

    def test_scale_invariant(self):
        """缩放不改变 cosine 距离 —— 这是 cosine 存在的全部理由。"""
        m = make_metric(METRIC_COSINE)
        a = as_array([1.0, 1.0] + [0.0] * 6)
        b = as_array([1.0, 2.0] + [0.0] * 6)
        b_big = as_array([100.0, 200.0] + [0.0] * 6)
        assert m.distance(a, 2.0, b, 5.0) == pytest.approx(
            m.distance(a, 2.0, b_big, squared_norm(b_big))
        )

    def test_store_normalizes_on_write(self):
        """入库向量必须被归一化，否则 cosine 退化成"看谁模长大"。"""
        store = make_store(metric=METRIC_COSINE)
        store.add([3.0, 4.0] + [0.0] * 6)
        assert squared_norm(store.values_of(0)) == pytest.approx(1.0)

    def test_store_normalizes_query_too(self):
        """查询侧与入库侧必须走同一条路。

        经典的错法是"库里归一化、查询不归一化"：距离会算出大于 1
        的荒谬值，而且看不出异常，只是结果"有点不对"。
        """
        store = make_store(metric=METRIC_COSINE)
        store.add([1.0] + [0.0] * 7)
        values, norm_sq = store.prepare_query([9.0] + [0.0] * 7)
        assert squared_norm(values) == pytest.approx(1.0)
        assert norm_sq == pytest.approx(1.0)

    def test_l2_store_does_not_normalize(self):
        """L2 度量下归一化会直接毁掉距离含义，绝不能顺手也做。"""
        store = make_store(metric=METRIC_L2)
        store.add([3.0, 4.0] + [0.0] * 6)
        assert squared_norm(store.values_of(0)) == pytest.approx(25.0)


class TestInnerProductDistance:
    def test_larger_inner_product_is_closer(self):
        """相似度越大越近，距离侧表现为越小。"""
        m = make_metric(METRIC_INNER_PRODUCT)
        pad = [0.0] * 6
        a = as_array([1.0, 0.0] + pad)
        near = as_array([1.0, 0.1] + pad)
        far = as_array([-1.0, 0.0] + pad)
        d_near = m.distance(a, 1.0, near, 1.01)
        d_far = m.distance(a, 1.0, far, 1.0)
        assert d_near < d_far

    def test_similarity_recovers_inner_product(self):
        m = make_metric(METRIC_INNER_PRODUCT)
        assert m.similarity_of(-3.5) == pytest.approx(3.5)


class TestVectorStore:
    def test_keys_are_sequential_from_zero(self):
        store = make_store()
        keys = add_all(store, [[1.0] * 8, [2.0] * 8, [3.0] * 8])
        assert keys == [0, 1, 2]

    def test_dimension_mismatch_rejected(self):
        store = make_store(dim=4)
        with pytest.raises(ValueError, match="维度不匹配"):
            store.add([1.0, 2.0])

    def test_zero_dimension_rejected(self):
        with pytest.raises(ValueError, match="维度必须是正整数"):
            VectorStore(0, make_metric(METRIC_L2))

    def test_non_finite_rejected(self):
        store = make_store()
        for bad in (float("inf"), float("nan"), float("-inf")):
            with pytest.raises(ValueError, match="inf 或 nan"):
                store.add([bad] * 8)

    def test_replace_updates_vector_and_norm_together(self):
        """replace 必须同时更新向量与模长。

        只换一个就是"距离悄悄算错"这类 bug 的标准入口，
        而且它在单元测试里几乎不会暴露（直到召回率莫名下降）。
        """
        store = make_store()
        store.add([3.0, 4.0] + [0.0] * 6)
        store.replace(0, [0.0, 0.0, 6.0, 8.0] + [0.0] * 4)
        assert squared_norm(store.values_of(0)) == pytest.approx(100.0)

    def test_replace_rejects_unknown_key(self):
        store = make_store()
        with pytest.raises(KeyError):
            store.replace(42, [0.0] * 8)

    def test_no_direct_mutation_api_exists(self):
        """刻意不提供改向量的方法。"""
        store = make_store()
        for forbidden in ("update", "set", "__setitem__", "pop", "remove"):
            assert not hasattr(store, forbidden), forbidden

    def test_revision_increments_on_write(self):
        store = make_store()
        assert store.revision == 0
        store.add([1.0] * 8)
        assert store.revision == 1
        store.add([2.0] * 8)
        assert store.revision == 2
        store.replace(0, [3.0] * 8)
        assert store.revision == 3

    def test_revision_does_not_move_on_read(self):
        """读操作不能碰 revision，否则索引会把自己判成过期。"""
        store = make_store()
        store.add([1.0] * 8)
        before = store.revision
        store.get(0)
        store.items()
        store.stats()
        store.memory_bytes()
        assert store.revision == before

    def test_label_roundtrip(self):
        store = make_store()
        store.add([1.0] * 8, label="doc-7")
        assert store.label_of(0) == "doc-7"

    def test_memory_uses_float32_array(self):
        """内存数字要有意义：不能把 Python 对象头算进去当压缩率。"""
        store = make_store(dim=32)
        store.add([0.0] * 32)
        # 32 维 float32 = 128 字节；list 存法要 600+ 字节
        assert store.memory_bytes() == 128


class TestTopK:
    def test_returns_smallest_k(self):
        scores = [(3.0, 0), (1.0, 1), (2.0, 2), (5.0, 3), (4.0, 4)]
        assert top_k(scores, 3) == [(1.0, 1), (2.0, 2), (3.0, 0)]

    def test_k_larger_than_input_returns_all(self):
        assert top_k([(1.0, 0)], 5) == [(1.0, 0)]

    def test_zero_k_returns_empty(self):
        assert top_k([(1.0, 0)], 0) == []

    def test_negative_k_rejected(self):
        with pytest.raises(ValueError):
            top_k([(1.0, 0)], -1)

    def test_stable_on_ties(self):
        """同一距离的多个 key，顺序必须可复现。"""
        scores = [(1.0, 9), (1.0, 3), (1.0, 7)]
        first = top_k(scores, 2)
        second = top_k(scores, 2)
        assert first == second


class TestGridVectors:
    def test_grid_size_and_bounds(self):
        store = make_store(dim=3)
        grid = grid_vectors(store, side=3)
        assert len(grid) == 27
        for v in grid:
            assert all(0.0 <= x <= 1.0 for x in v)
