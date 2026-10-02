"""IVF-PQ 的测试。

IVF-PQ 有两个和 HNSW 完全不同的失败模式，测试的重点也因此不同：
一是**信息丢失**（量化本身造成，调参救不回来），
二是**距离口径**（内部算的是 L2，要换算到外部度量）。
这两处的错都不报错，只表现为召回率低于预期。
"""

from __future__ import annotations

import pytest

from app.bruteforce import exact_top_k, recall_at_k, threshold_recall_at_k
from app.ivfpq import IVFPQConfig, IVFPQIndex, IVFPQStats, kmeans
from app.metrics import (
    METRIC_INNER_PRODUCT,
    VectorStore,
    make_metric,
    squared_norm,
)

DIM = 16


def build_index(n=200, dim=DIM, metric="l2", data_seed=1, **cfg):
    import random

    rng = random.Random(data_seed)
    metric_obj = make_metric(metric)
    store = VectorStore(dim, metric_obj)
    # 带簇结构的数据：PQ 在均匀数据上退化得很快，
    # 用簇结构才能测出"还剩下多少能力"
    centers = [[rng.gauss(0, 4.0) for _ in range(dim)] for _ in range(6)]
    for i in range(n):
        center = centers[i % 6]
        store.add([c + rng.gauss(0, 0.3) for c in center])
    index = IVFPQIndex(store, metric_obj, IVFPQConfig(**cfg))
    index.build()
    return store, metric_obj, index


class TestConfig:
    def test_nbits_above_eight_rejected(self):
        """码字按单字节存，> 8 位放不下。"""
        with pytest.raises(ValueError, match="nbits"):
            IVFPQConfig(nbits=9)

    def test_nprobe_zero_rejected(self):
        with pytest.raises(ValueError, match="nprobe"):
            IVFPQConfig(nprobe=0)

    def test_nlist_zero_rejected(self):
        with pytest.raises(ValueError, match="nlist"):
            IVFPQConfig(nlist=0)

    def test_codebook_size_is_two_pow_nbits(self):
        assert IVFPQConfig(nbits=4).codebook_size == 16
        assert IVFPQConfig(nbits=8).codebook_size == 256

    def test_sub_dim_divides_evenly(self):
        assert IVFPQConfig(m=8).sub_dim(32) == 4

    def test_dim_not_divisible_by_m_rejected(self):
        with pytest.raises(ValueError, match="整除"):
            IVFPQConfig(m=5).validate_dim(16)

    def test_dim_divisible_accepted(self):
        IVFPQConfig(m=8).validate_dim(16)

    def test_constructor_validates_dim(self):
        store = VectorStore(15, make_metric("l2"))
        store.add([0.0] * 15)
        with pytest.raises(ValueError, match="整除"):
            IVFPQIndex(store, make_metric("l2"), IVFPQConfig(m=8))


class TestMetricSupport:
    def test_inner_product_rejected_at_construction(self):
        """内积必须与 IVF-PQ 绝缘。

        内部按 L2 量化，内积没有可换算的恒等式；
        放任它跑会给出一组"看起来能用"但排序毫无意义的结果。
        """
        store = VectorStore(DIM, make_metric(METRIC_INNER_PRODUCT))
        store.add([1.0] * DIM)
        with pytest.raises(ValueError, match="内积"):
            IVFPQIndex(store, make_metric(METRIC_INNER_PRODUCT))

    def test_error_message_points_at_cosine(self):
        store = VectorStore(DIM, make_metric(METRIC_INNER_PRODUCT))
        store.add([1.0] * DIM)
        with pytest.raises(ValueError, match="cosine"):
            IVFPQIndex(store, make_metric(METRIC_INNER_PRODUCT))

    def test_l2_and_cosine_accepted(self):
        for name in ("l2", "cosine"):
            store = VectorStore(DIM, make_metric(name))
            store.add([1.0] * DIM)
            IVFPQIndex(store, make_metric(name))


class TestKMeans:
    """k-means 的两个必须做对的地方：k-means++ 初始化、空簇重置。

    返回的是质心列表（不含分配表）—— IVF 只关心质心，
    分配在 build 里现算。少返回一个值就意味着少一处可能不同步的状态。
    """

    def test_recovers_two_well_separated_clusters(self):
        samples = [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 9.9]]
        centroids = kmeans(samples, k=2, dim=2, iterations=10, seed=1)
        assert len(centroids) == 2
        assert all(len(c) == 2 for c in centroids)
        # 两个质心应当分别落在两组附近：一个靠近原点，一个靠近 (10,10)
        near_origin = min(abs(c[0]) + abs(c[1]) for c in centroids)
        near_far = max(abs(c[0]) + abs(c[1]) for c in centroids)
        assert near_origin < 1.0
        assert near_far > 15.0

    def test_no_empty_clusters_when_points_outnumber_clusters(self):
        """空簇是 k-means 里必须显式处理的分支。

        空簇永不分到点，等于永久浪费一个聚类名额；
        而它不报错，只是倒排表里有一列永远没人用。

        这里的检验方式：每个质心都应该落在"有数据的区域" ——
        如果某个质心被留在一个没人占用的位置上，
        它到最近样本的距离会明显大于其他质心。
        """
        samples = [[0.0, 0.0]] * 20 + [[5.0, 5.0]] * 20
        centroids = kmeans(samples, k=4, dim=2, iterations=6, seed=2)
        assert len(centroids) == 4
        nearest = [
            min(abs(c[0] - s[0]) + abs(c[1] - s[1]) for s in samples)
            for c in centroids
        ]
        assert max(nearest) < 2.0

    def test_fewer_points_than_clusters_is_clamped(self):
        """点比簇还少时**夹取**到样本数。

        凭空造不存在的簇没有意义；但要注意后果：IVF 的倒排表数量
        取决于实际质心数，所以这时 nlist 会小于请求值。
        文档曾经写成"会得到一些空簇"，与实现对不上 ——
        这条测试把正确行为钉住。
        """
        samples = [[1.0, 1.0], [2.0, 2.0]]
        centroids = kmeans(samples, k=5, dim=2, iterations=3, seed=3)
        assert len(centroids) == 2

    def test_single_cluster_lands_near_data_mean(self):
        samples = [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]]
        centroids = kmeans(samples, k=1, dim=2, iterations=4, seed=4)
        assert len(centroids) == 1
        assert centroids[0][0] == pytest.approx(2.0, abs=0.5)

    def test_zero_iterations_still_returns_non_degenerate_centroids(self):
        """iterations=0 时质心来自 k-means++ 初始化，不能是零向量。

        如果初始化退化成"取前 k 个零向量"，所有粗聚类都会失效，
        而且错误信号是"召回率极低"而不是抛异常。
        """
        samples = [[float(i) + 1.0, 0.0] for i in range(8)]
        centroids = kmeans(samples, k=2, dim=2, iterations=0, seed=5)
        assert len(centroids) == 2
        assert all(c[0] > 0.0 for c in centroids)

    def test_zero_k_rejected(self):
        with pytest.raises(ValueError, match="k 必须为正整数"):
            kmeans([[1.0, 1.0]], k=0, dim=2)

    def test_empty_samples_rejected(self):
        with pytest.raises(ValueError, match="没有样本"):
            kmeans([], k=2, dim=2)

    def test_deterministic_under_fixed_seed(self):
        samples = [[float(i % 7), float(i % 3)] for i in range(30)]
        a = kmeans(samples, k=3, dim=2, iterations=5, seed=9)
        b = kmeans(samples, k=3, dim=2, iterations=5, seed=9)
        assert [list(c) for c in a] == [list(c) for c in b]


class TestBuild:
    def test_empty_store_rejected(self):
        store = VectorStore(DIM, make_metric("l2"))
        index = IVFPQIndex(store, make_metric("l2"))
        with pytest.raises(ValueError, match="空的"):
            index.build()

    def test_lists_cover_every_vector(self):
        store, _, index = build_index(n=90)
        total = sum(len(keys) for keys in index._keys)
        assert total == 90

    def test_codes_are_bytes(self):
        """码必须按单字节存 —— 这是压缩率账的基础。"""
        store, _, index = build_index(n=40)
        for per_list in index._codes:
            for code in per_list:
                assert code.typecode == "B"
                assert len(code) == index.config.m

    def test_codebooks_match_sub_dimension(self):
        store, _, index = build_index(n=40)
        assert len(index._codebooks) == index.config.m
        for sub_codebook in index._codebooks:
            assert len(sub_codebook) == index.config.codebook_size
            for centroid in sub_codebook:
                assert len(centroid) == index.sub_dim

    def test_keys_and_codes_aligned(self):
        store, _, index = build_index(n=60)
        for keys, codes in zip(index._keys, index._codes):
            assert len(keys) == len(codes)

    def test_trained_equals_corpus_size(self):
        store, _, index = build_index(n=70)
        assert index.stats.trained == 70

    def test_build_is_deterministic(self):
        a = build_index(n=80, data_seed=7)[2]
        b = build_index(n=80, data_seed=7)[2]
        assert [list(c) for c in a._coarse] == [list(c) for c in b._coarse]

    def test_training_sample_limit_is_respected(self):
        """采样用等距抽样而不是随机抽样。

        随机抽样会让"固定 seed"这个承诺依赖抽样序列的长度，
        改一个别的参数（间接改变 RNG 调用次数）就会让结果全变。
        """
        store, _, index = build_index(n=40, train_sample_limit=20)
        assert index.stats.train_seconds >= 0.0


class TestSearch:
    def test_returns_k_results(self):
        store, metric, index = build_index(n=150, nprobe=4)
        q = store.prepare_query(list(store.values_of(0)))
        assert len(index.search(q[0], q[1], k=5)) == 5

    def test_results_sorted_ascending(self):
        store, metric, index = build_index(n=150, nprobe=4)
        q = store.prepare_query(list(store.values_of(0)))
        hits = index.search(q[0], q[1], k=5)
        assert [d for d, _ in hits] == sorted(d for d, _ in hits)

    def test_finds_the_query_point_itself(self):
        """查询点自己入库了，量化再糙也不该把自己丢了。"""
        store, metric, index = build_index(n=150, nprobe=8, nbits=8)
        q = store.prepare_query(list(store.values_of(3)))
        hits = index.search(q[0], q[1], k=1)
        assert hits[0][1] == 3

    def test_nprobe_is_a_monotone_knob(self):
        """nprobe 越大召回率不降 —— 这是它作为旋钮的前提。"""
        store, metric, index = build_index(n=200, nprobe=1)
        low = _mean_recall(store, metric, index, nprobe=1)
        high = _mean_recall(store, metric, index, nprobe=index.config.nlist)
        assert high >= low

    def test_zero_k_returns_empty(self):
        store, metric, index = build_index(n=100)
        q = store.prepare_query(list(store.values_of(0)))
        assert index.search(q[0], q[1], k=0) == []

    def test_search_before_build_raises(self):
        store = VectorStore(DIM, make_metric("l2"))
        for i in range(20):
            store.add([float(i)] * DIM)
        index = IVFPQIndex(store, make_metric("l2"))
        q = store.prepare_query([1.0] * DIM)
        with pytest.raises(RuntimeError):
            index.search(q[0], q[1], k=3)

    def test_query_is_idempotent(self):
        store, metric, index = build_index(n=120)
        q = store.prepare_query(list(store.values_of(5)))
        first = index.search(q[0], q[1], k=5)
        second = index.search(q[0], q[1], k=5)
        assert first == second

    def test_scanned_count_reported(self):
        store, metric, index = build_index(n=200, nprobe=2)
        q = store.prepare_query(list(store.values_of(2)))
        index.search(q[0], q[1], k=5)
        assert index.stats.scanned_total > 0
        assert index.query_invocations >= 1

    def test_high_nprobe_scans_more_than_low(self):
        store, metric, index = build_index(n=200)
        q = store.prepare_query(list(store.values_of(2)))
        index.search(q[0], q[1], k=5, nprobe=1)
        low = index.stats.scanned_total / index.query_invocations
        index.search(q[0], q[1], k=5, nprobe=index.config.nlist)
        high = index.stats.scanned_total / index.query_invocations
        assert high >= low


class TestStaleIndex:
    def test_adding_vector_after_build_raises(self):
        from app.hnsw import StaleIndexError

        store, metric, index = build_index(n=80)
        store.add([1.0] * DIM)
        q = store.prepare_query([1.0] * DIM)
        with pytest.raises(StaleIndexError):
            index.search(q[0], q[1], k=3)

    def test_rebuild_restores_usability(self):
        store, metric, index = build_index(n=80)
        store.add([1.0] * DIM)
        index.build()
        q = store.prepare_query([1.0] * DIM)
        assert len(index.search(q[0], q[1], k=3)) == 3


class TestQuantisationLoss:
    """量化损失：IVF-PQ 的召回率上限由压缩率决定。

    这一组不是"测试"，是**把性能扶贫等级量化钉住**的回归哨兵：
    哪天有人改了码本训练，"nbits 越高召回率越高"这个单调关系坏了，
    这里会第一个报错。
    """

    def test_more_bits_gives_better_recall(self):
        store, metric, index = build_index(n=200, m=8, nbits=8)
        index.build()
        good = _mean_recall(store, metric, index, nprobe=index.config.nlist)

        store2, metric2, coarse_index = build_index(n=200, m=8, nbits=4)
        weak = _mean_recall(store2, metric2, coarse_index,
                            nprobe=coarse_index.config.nlist)
        assert good >= weak

    def test_more_subspaces_gives_better_recall(self):
        _, _, fine = build_index(n=200, m=16, nbits=4)
        fine_recall = _mean_recall(fine.store, fine.metric, fine,
                                   nprobe=fine.config.nlist)
        _, _, coarse = build_index(n=200, m=8, nbits=4)
        coarse_recall = _mean_recall(coarse.store, coarse.metric, coarse,
                                     nprobe=coarse.config.nlist)
        assert fine_recall >= coarse_recall

    def test_threshold_recall_exceeds_standard_on_ties(self):
        """并列多时门槛召回率会显著高于标准召回率。

        PQ 的近似距离让大量候选在"精确距离的门槛"附近扎堆 ——
        这时候两个召回率拉开差距是正常的，不代表实现有错。
        """
        store, metric, index = build_index(n=200, m=8, nbits=4)
        std_total = 0.0
        thr_total = 0.0
        trials = 10
        for i in range(trials):
            q = store.prepare_query(list(store.values_of(i)))
            exact = exact_top_k(store, q, metric, k=10)
            hits = index.search(q[0], q[1], k=10, nprobe=index.config.nlist)
            std_total += recall_at_k([k for _, k in hits], exact)
            thr_total += threshold_recall_at_k(hits, exact)
        assert thr_total >= std_total


class TestDistanceConversion:
    def test_l2_needs_no_conversion(self):
        store, metric, index = build_index(n=60)
        assert index._to_metric_distance(4.0) == pytest.approx(4.0)

    def test_cosine_is_exactly_half_squared_distance(self):
        """归一化后 ‖a-b‖² = 2 - 2cos，故 cos 距离 = 1-cos = ‖a-b‖²/2。

        这个恒等式是**精确的**，不是近似：正因为如此
        IVF-PQ 才能在 cosine 度量下复用 L2 量化而只加最后一记换算。
        若有一日这里变成近似，下面的断言会立刻红。
        """
        store, metric, index = build_index(n=60, metric="cosine", m=8, nbits=4)
        for l2_sq in (0.0, 0.25, 1.0, 2.0):
            assert index._to_metric_distance(l2_sq) == pytest.approx(l2_sq / 2)

    def test_cosine_distance_matches_manual_cosine(self):
        import math

        store = VectorStore(4, make_metric("cosine"))
        store.add([1.0, 0.0, 0.0, 0.0])
        store.add([0.0, 1.0, 0.0, 0.0])
        a = list(store.values_of(0))
        b = list(store.values_of(1))
        manual_cos = sum(x * y for x, y in zip(a, b)) / math.sqrt(
            sum(x * x for x in a) * sum(y * y for y in b)
        )
        index = IVFPQIndex(store, make_metric("cosine"), IVFPQConfig(m=2))
        l2_sq = sum((x - y) ** 2 for x, y in zip(a, b))
        assert index._to_metric_distance(l2_sq) == pytest.approx(1 - manual_cos)


class TestMemoryAccounting:
    def test_compression_ratio_is_positive(self):
        store, _, index = build_index(n=120)
        assert index.memory_bytes()["compression_ratio"] > 1.0

    def test_codes_only_compression_equals_theory(self):
        """纯编码口径必须等于 4d/m —— 理论值。

        total 口径会被码本/倒排表这些与 n 无关的开销稀释，
        n 小时远达不到这个值。两个数字分开列，
        就是为了避免拿一个口径去否定另一个。
        """
        store, _, index = build_index(n=120, m=8)
        mem = index.memory_bytes()
        assert mem["codes_only_compression"] == pytest.approx(4 * DIM / 8)

    def test_total_includes_every_component(self):
        store, _, index = build_index(n=120)
        mem = index.memory_bytes()
        assert mem["total"] == (
            mem["compressed_codes"]
            + mem["codebooks"]
            + mem["coarse_centroids"]
            + mem["inverted_lists"]
        )

    def test_codebooks_are_the_fixed_overhead(self):
        """码本与向量数量无关 —— 这就是小数据集压缩比难看的原因。"""
        small, _, small_index = build_index(n=60)
        large, _, large_index = build_index(n=180)
        assert small_index.memory_bytes()["codebooks"] == (
            large_index.memory_bytes()["codebooks"]
        )

    def test_does_not_store_vectors(self):
        """memory_bytes 里不能有 vectors —— 索引不保存原始向量。"""
        store, _, index = build_index(n=80)
        assert "vectors" not in index.memory_bytes()


class TestSnapshot:
    def test_snapshot_contract(self):
        store, _, index = build_index(n=120)
        snap = index.snapshot()
        for name in (
            "trained", "nlist", "probed_lists", "list_sizes",
            "memory", "config", "build_seconds", "avg_scanned_per_query",
        ):
            assert name in snap, name

    def test_list_sizes_reports_empty_lists(self):
        store, _, index = build_index(n=120, nlist=16)
        sizes = index.snapshot()["list_sizes"]
        assert sizes["min"] >= 0
        assert sizes["max"] >= sizes["min"]
        assert 0 <= sizes["empty"] <= index.config.nlist

    def test_avg_scanned_is_positive(self):
        store, _, index = build_index(n=120)
        q = store.prepare_query(list(store.values_of(0)))
        index.search(q[0], q[1], k=3)
        assert index.snapshot()["avg_scanned_per_query"] > 0


def _mean_recall(store, metric, index, nprobe, trials=10, k=10):
    total = 0.0
    for i in range(trials):
        q = store.prepare_query(list(store.values_of(i)))
        exact = exact_top_k(store, q, metric, k=k)
        hits = index.search(q[0], q[1], k=k, nprobe=nprobe)
        total += recall_at_k([key for _, key in hits], exact)
    return total / trials
