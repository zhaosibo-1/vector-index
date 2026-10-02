"""数据集生成器的测试。

生成器是**所有召回率数字的地基**：如果生成的数据结构太理想
（比如簇完全分离），ANN 召回率会虚高，测不出索引的真实能力；
如果不理想到全是噪声，又什么都测不出来。这里断言的是
"结构确实存在、但不送分"。
"""

from __future__ import annotations

import pytest

from app.dataset import (
    MAX_COUNT,
    MAX_DIM,
    MIN_COUNT,
    VALID_KINDS,
    Dataset,
    generate,
    generate_queries,
)
from app.metrics import VectorStore, make_metric, squared_norm


class TestBounds:
    def test_count_too_small_rejected(self):
        with pytest.raises(ValueError, match="count"):
            generate(count=MIN_COUNT - 1, dim=4)

    def test_count_too_large_rejected(self):
        # 上限是硬拒绝而不是"跑得慢"：超时的表现是 504，
        # 用户完全看不出是自己参数太大
        with pytest.raises(ValueError, match="count"):
            generate(count=MAX_COUNT + 1, dim=4)

    def test_dim_too_small_rejected(self):
        with pytest.raises(ValueError, match="dim"):
            generate(count=30, dim=1)

    def test_dim_too_large_rejected(self):
        with pytest.raises(ValueError, match="dim"):
            generate(count=30, dim=MAX_DIM + 1)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="数据集类型"):
            generate(count=30, dim=4, kind="spiral")

    def test_all_valid_kinds_accepted(self):
        for kind in VALID_KINDS:
            ds = generate(count=30, dim=4, kind=kind)
            assert ds.kind == kind
            assert len(ds.vectors) == 30


class TestShape:
    def test_vector_dimensions_match(self):
        ds = generate(count=40, dim=6, kind="uniform")
        assert all(len(v) == 6 for v in ds.vectors)

    def test_all_values_finite(self):
        ds = generate(count=40, dim=6, kind="gaussian")
        import math

        for v in ds.vectors:
            assert all(math.isfinite(x) for x in v)

    def test_labels_present_when_requested(self):
        ds = generate(count=30, dim=4)
        assert len(ds.labels) == 30
        assert ds.labels[0] != ds.labels[1]

    def test_params_recorded(self):
        """生成参数必须回显，否则 benchmark 数字无法复现。"""
        ds = generate(count=30, dim=4, kind="clustered", n_clusters=5, seed=11)
        assert ds.params["n_clusters"] == 5
        assert ds.seed == 11


class TestDeterminism:
    def test_same_seed_gives_same_vectors(self):
        a = generate(count=60, dim=5, kind="clustered", seed=99)
        b = generate(count=60, dim=5, kind="clustered", seed=99)
        assert [list(v) for v in a.vectors] == [list(v) for v in b.vectors]

    def test_different_seed_gives_different_vectors(self):
        a = generate(count=60, dim=5, kind="clustered", seed=99)
        b = generate(count=60, dim=5, kind="clustered", seed=100)
        assert [list(v) for v in a.vectors] != [list(v) for v in b.vectors]

    def test_same_seed_gives_same_queries(self):
        ds = generate(count=60, dim=5, kind="clustered", seed=99)
        q1 = generate_queries(ds, n_queries=7, seed=5)
        q2 = generate_queries(ds, n_queries=7, seed=5)
        assert [list(v) for v in q1] == [list(v) for v in q2]

    def test_never_touches_global_random(self):
        """必须用私有 Random 实例。

        用全局 ``random`` 会让"seed 相同"这个承诺失效 ——
        别的测试先调一次 random 就会改变下一个数据集。
        """
        import random

        random.seed(1234)
        before = random.random()
        random.seed(1234)
        generate(count=30, dim=4, kind="clustered", seed=7)
        after = random.random()
        assert before == after


class TestStructure:
    def test_clustered_data_has_real_structure(self):
        """簇内距离必须显著小于跨簇距离。

        这条是整个 benchmark 有意义的前提：如果"有结构"的数据
        和均匀噪声统计上没区别，那 recall 数字说明不了任何事。

        构造顺序是 ``i % n_clusters``，所以 key 0/8/16 属于同一簇 ——
        **用 label 而不是"前两个点"来分组**，后者在簇交错排列下
        根本不是一回事（最开始写这个测试时就踩了这个坑）。
        """
        dim = 8
        n_clusters = 8
        ds = generate(count=200, dim=dim, kind="clustered",
                      n_clusters=n_clusters, seed=3)
        store = VectorStore(dim, make_metric("l2"))
        for v in ds.vectors:
            store.add(v)

        same = [0, n_clusters, 2 * n_clusters]      # 全在 c0
        cross = [0, 1, 2]                            # c0 / c1 / c2 各一个
        assert _pairwise_mean(store, same) < _pairwise_mean(store, cross)
        assert _pairwise_mean(store, same) < _pairwise_mean(store, cross) * 0.5

    def test_labels_carry_cluster_id(self):
        """簇标号要写在 label 里，否则没法验证"邻居是否同簇"。"""
        ds = generate(count=40, dim=4, kind="clustered", n_clusters=4, seed=3)
        assert ds.labels[0].endswith("@c0")
        assert ds.labels[4].endswith("@c0")
        assert ds.labels[1].endswith("@c1")

    def test_overlap_makes_clusters_dirtier(self):
        """overlap 越大，簇越糊 —— 调它应当真的改变数据难度。"""
        clean = generate(
            count=200, dim=8, kind="clustered", n_clusters=8,
            overlap=0.02, seed=3,
        )
        dirty = generate(
            count=200, dim=8, kind="clustered", n_clusters=8,
            overlap=0.6, seed=3,
        )
        store_a = VectorStore(8, make_metric("l2"))
        store_b = VectorStore(8, make_metric("l2"))
        for v in clean.vectors:
            store_a.add(v)
        for v in dirty.vectors:
            store_b.add(v)
        # 脏数据的最近邻距离应当更大（邻居更远了）
        assert _nn_distance(store_b, 0) > _nn_distance(store_a, 0)

    def test_duplicates_mode_produces_near_copies(self):
        """duplicates 模式必须产生极近的副本（不是精确相等）。

        扰动量是 ``overlap × 0.01``，远小于簇内散度 —— 于是它们会
        互相成为最近邻，制造出**大量并列**。这一档数据就是专门用来
        测并列时的 tie-break 与召回率分母的（见 bruteforce.py）。

        刻意**不做**精确复制：相等的点在余弦度量下完全无法区分，
        而 L2 下又会因为零距离让排序退化，两种情况都测不出
        "近似相等但不相等"这个真实场景。
        """
        dim = 6
        ds = generate(count=120, dim=dim, kind="duplicates",
                      duplicate_ratio=0.5, seed=5, n_clusters=6, overlap=0.15)
        store = VectorStore(dim, make_metric("l2"))
        for v in ds.vectors:
            store.add(v)

        n_dupes = ds.params["duplicate_count"]
        # key i 复制自 key n_dupes + i，两者相距只有 jitter 量级
        near = _pairwise_mean(store, [0, n_dupes])
        far = _pairwise_mean(store, [0, dim])
        assert near < far * 0.05
        assert near > 0.0

    def test_duplicate_ratio_zero_means_no_copies(self):
        """ratio=0 时不能偷偷混入副本。"""
        ds = generate(count=60, dim=4, kind="duplicates",
                      duplicate_ratio=0.0, seed=5)
        assert ds.params.get("duplicate_count", 0) == 0
        assert not any(l.endswith("~dup") for l in ds.labels)

    def test_duplicate_pairs_are_disjoint_from_sources(self):
        """副本的来源区间与写入区间不能重叠。

        重叠就意味着"一部分点在复制另一部分还在被读的点"，
        顺序一变结果就变。这条断言的是当初那个 bug 的结构性条件。
        """
        ds = generate(count=100, dim=4, kind="duplicates",
                      duplicate_ratio=0.4, seed=5)
        n_dupes = ds.params["duplicate_count"]
        assert 0 < n_dupes < 100
        dupes = {i for i, l in enumerate(ds.labels) if l.endswith("~dup")}
        assert dupes == set(range(n_dupes))
        for i in dupes:
            assert not ds.labels[n_dupes + i].endswith("~dup")

    def test_duplicate_label_points_at_source_cluster(self):
        """副本的 label 要写来源簇号，不能写自己排第几位。"""
        ds = generate(count=100, dim=4, kind="duplicates",
                      duplicate_ratio=0.4, seed=5, n_clusters=5)
        n_dupes = ds.params["duplicate_count"]
        assert ds.labels[0].endswith("~dup")
        # 来源是 key n_dupes，其簇号 = n_dupes % 5
        assert ds.labels[0] == f"vec-00000@c{n_dupes % 5}~dup"

    def test_uniform_fills_unit_range(self):
        ds = generate(count=80, dim=4, kind="uniform")
        for v in ds.vectors:
            assert all(0.0 <= x <= 1.0 for x in v)


class TestQueries:
    def test_queries_come_from_the_dataset(self):
        """查询要从数据分布里取，否则召回率虚高。

        分布外的查询，其最近邻距离会比库内任意两点都远得多，
        ANN 反而更容易做对 —— 那时候数字好看但说明不了问题。
        """
        ds = generate(count=200, dim=8, kind="clustered", seed=3)
        store = VectorStore(8, make_metric("l2"))
        for v in ds.vectors:
            store.add(v)
        queries = generate_queries(ds, n_queries=10, seed=4)
        assert len(queries) == 10
        for q in queries:
            assert len(q) == 8
        # 每个查询都应当在库里有一个不算太远的邻居
        metric = make_metric("l2")
        for q in queries:
            best = min(
                metric.distance(q, squared_norm(q), store.values_of(k),
                                squared_norm(store.values_of(k)))
                for k in store.all_keys()
            )
            assert best < 1.0

    def test_zero_queries_rejected(self):
        """n_queries=0 是调用方算错下标，不能静默返回空列表。

        静默返回空会让 benchmark 跑出一堆 0 或者除零，
        而没人会怀疑是查询数量传错了。
        """
        ds = generate(count=30, dim=4)
        with pytest.raises(ValueError, match="n_queries"):
            generate_queries(ds, n_queries=0)

    def test_queries_are_a_superset_check(self):
        ds = generate(count=30, dim=4)
        qs = generate_queries(ds, n_queries=5)
        assert all(isinstance(q, type(ds.vectors[0])) for q in qs)


def _pairwise_mean(store: VectorStore, keys: list[int]) -> float:
    metric = make_metric("l2")
    total = 0.0
    count = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            total += metric.distance(
                store.values_of(a), squared_norm(store.values_of(a)),
                store.values_of(b), squared_norm(store.values_of(b)),
            )
            count += 1
    return total / count


def _nn_distance(store: VectorStore, key: int) -> float:
    metric = make_metric("l2")
    target = store.values_of(key)
    best = float("inf")
    for other in store.all_keys():
        if other == key:
            continue
        best = min(
            best,
            metric.distance(
                target, squared_norm(target),
                store.values_of(other), squared_norm(store.values_of(other)),
            ),
        )
    return best


def test_dataset_is_frozen_ish():
    """Dataset 不是 frozen（vectors 是可变的），但字段类型要对。"""
    ds = generate(count=30, dim=4)
    assert isinstance(ds, Dataset)
