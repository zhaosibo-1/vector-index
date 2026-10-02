"""合成数据集：让"召回率"这个数字有意义。

==========================================================================
 为什么不能随便撒随机点
==========================================================================
在**均匀随机**的高维数据上，任意两点的距离几乎相同（测度集中现象：
d 维单位超立方体里，最大距离与最小距离的比值随 d 趋于 1）。
这带来两个后果：

* 精确 kNN 的"最近邻"本身就没有意义 —— 它和第 100 名邻居只差万分之一；
* 所有近似索引的召回率都会**惨不忍睹**，不是算法坏了，
  而是"近邻结构"根本不存在，任何图或倒排都抓不住东西。

所以数据集必须有**结构**。这个模块生成带簇结构的数据，
并且把"簇有多难分"做成一个显式参数（``overlap``），
这样召回率随参数变化的曲线才有解释力。

同时**保留**一个 ``uniform`` 模式，并在基准里对比 ——
让"均匀随机数据上 ANN 完全失效"这件事被量出来、写在 README 里，
而不是让读者自己踩一遍。

==========================================================================
 确定性
==========================================================================
同一个 ``seed`` 一定生成同一份数据。这不是锦上添花：
基准测试的所有结论都要可比，如果每次生成的数据不同，
"改了 ef_construction 之后召回率涨了 3%"这种结论就完全站不住
（可能只是这次的数据更好找邻居）。

实现上用 ``random.Random(seed)`` 的独立实例，**不碰全局 random**。
用全局的话，测试里别的代码调一次 ``random.random()`` 就会改变
本模块后续所有输出 —— 那是最难查的一类不可复现。
"""

from __future__ import annotations

import math
import random
from array import array
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 规模上限
# ---------------------------------------------------------------------------
# 纯 Python 的索引构建是 O(n · ef · log n) 级别的距离计算，实测一次距离 1.5µs。
# n=50000 时要几千万次比较，也就是好几分钟 —— 单个 HTTP 请求等不起，
# 而且会把事件循环彻底堵死。
# 所以这里设硬上限并在**入口**拒绝，而不是等它自己超时：
# 超时的表现是一个 504/连接断开，用户完全看不出是自己参数太大。
MAX_COUNT = 20_000
MAX_DIM = 512
MIN_COUNT = 20
MIN_DIM = 2

VALID_KINDS: tuple[str, ...] = ("clustered", "uniform", "gaussian", "duplicates")


@dataclass
class Dataset:
    """一份生成好的数据集。"""

    kind: str
    dim: int
    count: int
    metric: str
    seed: int
    vectors: list[array]
    labels: list[str] = field(default_factory=list)
    #: 生成参数（簇数、重叠度等），用于复现与展示
    params: dict[str, object] = field(default_factory=dict)

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "dim": self.dim,
            "count": self.count,
            "metric": self.metric,
            "seed": self.seed,
            **self.params,
        }


def _check_bounds(count: int, dim: int) -> None:
    if not (MIN_COUNT <= count <= MAX_COUNT):
        raise ValueError(
            f"count 必须在 [{MIN_COUNT}, {MAX_COUNT}] 内，收到 {count}"
        )
    if not (MIN_DIM <= dim <= MAX_DIM):
        raise ValueError(f"dim 必须在 [{MIN_DIM}, {MAX_DIM}] 内，收到 {dim}")


def _random_unit_vector(dim: int, rng: random.Random) -> array:
    return array("f", (rng.gauss(0.0, 1.0) for _ in range(dim)))


def _normalize(values: array) -> array:
    norm = math.sqrt(math.sumprod(values, values))
    if norm < 1e-12:
        # 理论上 gauss 生不出全零向量，但既然有 if，就把它写对：
        # 返回一个确定的单位向量而不是除零得到 inf/nan。
        fallback = array("f", [0.0] * len(values))
        fallback[0] = 1.0
        return fallback
    return array("f", (v / norm for v in values))


def _build_centroids(n_clusters: int, dim: int, rng: random.Random) -> list[array]:
    """簇心取单位球面上的随机方向。

    用方向而不是超立方体里的点：超立方体的角点距离分布非常不均匀，
    会让"簇间距离"有很大的随机波动，同一组参数两次生成的难度差很多。
    单位球面上的方向分布均匀，簇间距离稳定，召回率曲线才平滑可解释。
    """
    centroids: list[array] = []
    for _ in range(n_clusters):
        centroids.append(_normalize(_random_unit_vector(dim, rng)))
    return centroids


def generate(
    *,
    count: int = 2000,
    dim: int = 32,
    kind: str = "clustered",
    metric: str = "l2",
    seed: int = 42,
    n_clusters: int = 24,
    #: 簇内散度。相对簇间距离的比值越小，簇越"干净"、ANN 越好做。
    #: 0.15 大致是"有明显结构但不送分"的量级，也是默认基准用的值。
    overlap: float = 0.15,
    #: duplicates 模式下，有多少比例的点是"另一个点的近复制"
    duplicate_ratio: float = 0.2,
) -> Dataset:
    """生成数据集。

    ``kind`` 的四种模式各有用途：

    * ``clustered``  —— 默认。带簇结构，ANN 能发挥作用，召回率曲线有意义。
    * ``uniform``    —— 均匀随机。**高维下近邻结构不存在**，
                       用作"ANN 失效"的对照，同时也是暴力检索最快的场景。
    * ``gaussian``   —— 单一高斯。比 uniform 稍微有点结构（中心密集），
                       介于两者之间。
    * ``duplicates`` —— 在 clustered 基础上混入近复制点。
                       用来验证索引在**有大量并列最近邻**时的行为：
                       tie-break 是否稳定、召回率的分母该怎么算。
                       （精确 kNN 遇到并列时取哪个，会直接影响
                       "召回率"这个指标的定义 —— 见 bruteforce.py）
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"未知的数据集类型 {kind!r}；合法值：{'、'.join(VALID_KINDS)}")
    _check_bounds(count, dim)

    rng = random.Random(seed)
    params: dict[str, object] = {}

    if kind in ("clustered", "duplicates"):
        if n_clusters < 1:
            raise ValueError(f"n_clusters 必须 ≥ 1，收到 {n_clusters}")
        if overlap <= 0:
            raise ValueError(f"overlap 必须为正数，收到 {overlap}")
        # 簇心数量不可能超过点数，否则会有空簇 —— k-means 里空簇是
        # 一个必须处理的分支，但在这里没有意义，直接夹取。
        effective_clusters = min(n_clusters, count)
        centroids = _build_centroids(effective_clusters, dim, rng)

        vectors: list[array] = []
        for i in range(count):
            centroid = centroids[i % effective_clusters]
            noise = array("f", (rng.gauss(0.0, overlap) for _ in range(dim)))
            vectors.append(array("f", (c + n for c, n in zip(centroid, noise))))

        params.update(
            {
                "n_clusters": effective_clusters,
                "overlap": overlap,
                "centroid_spread": 1.0,
            }
        )

        #: ``duplicate_source[i]`` = 第 i 个点是从谁复制来的（None 表示原生的）。
        #: 单独存一份而不是事后反推，是因为**簇标号 label 要靠它** ——
        #: 一个副本的身份是"它来源那一簇"，不是它自己排在第几位。
        duplicate_source: list[int | None] = [None] * count

        if kind == "duplicates":
            # 副本数量取 count//2 的上限：来源点必须是另一批点，
            # ratio=0.9 时不能把 90% 的点都变成另外 90% 的副本。
            n_dupes = min(int(count * duplicate_ratio), count // 2)
            for i in range(n_dupes):
                # 来源是 [n_dupes, 2·n_dupes) 这一段，与写入区间
                # [0, n_dupes) **不相交**。
                #
                # 这里曾经写成 ``source = vectors[i % n_dupes]``，
                # 而循环变量 i 本身就在 [0, n_dupes) 里 ——
                # 于是每个点都在"复制自己再加一点噪声"，
                # duplicates 模式静默退化成 clustered。
                # 它不报错、召回率也照样是 1.0，
                # 只有专门去看"副本对之间到底有多近"才会发现。
                source = vectors[n_dupes + i]
                # 复制 + 极小扰动：距离远小于簇内正常间距，
                # 于是它们会互相成为最近邻，制造出并列的情况。
                jitter = array(
                    "f", (rng.gauss(0.0, overlap * 0.01) for _ in range(dim))
                )
                vectors[i] = array("f", (s + j for s, j in zip(source, jitter)))
                duplicate_source[i] = n_dupes + i
            params["duplicate_ratio"] = duplicate_ratio
            params["duplicate_count"] = n_dupes

    elif kind == "uniform":
        # 落在单位超立方体里。注意**不做归一化**：
        # L2 场景下归一化会人为引入球面结构，那就不是"均匀"了。
        vectors = [
            array("f", (rng.random() for _ in range(dim))) for _ in range(count)
        ]
        params["range"] = [0.0, 1.0]

    else:  # gaussian
        vectors = []
        for _ in range(count):
            vectors.append(
                array("f", (rng.gauss(0.0, 1.0) for _ in range(dim)))
            )
        params["std"] = 1.0

    labels = [f"vec-{i:05d}" for i in range(count)]
    # 构造顺序里 i % n_clusters 决定了簇归属，把簇标号也带上，
    # 后面可以用来验证"近似检索找到的邻居是否来自同一个簇"。
    # duplicates 模式下副本归属**来源点**所在的簇。
    if kind in ("clustered", "duplicates"):
        n_clusters_actual = int(params["n_clusters"])
        labels = []
        for i in range(count):
            origin = duplicate_source[i] if duplicate_source[i] is not None else i
            tag = "~dup" if duplicate_source[i] is not None else ""
            labels.append(f"vec-{i:05d}@c{origin % n_clusters_actual}{tag}")

    return Dataset(
        kind=kind,
        dim=dim,
        count=count,
        metric=metric,
        seed=seed,
        vectors=vectors,
        labels=labels,
        params=params,
    )


def generate_queries(
    dataset: Dataset, *, n_queries: int = 50, seed: int | None = None
) -> list[array]:
    """生成查询向量。

    默认**从数据集里采样**已有的点（并加一点扰动），而不是重新撒点。
    这是检索评测的标准做法，原因很实际：

    如果查询点落在数据分布之外（真·随机撒点），那么"最近邻"的距离
    会比数据内部任意两点都远得多，最近的几个邻居之间的差距也会很大 ——
    这时候 ANN 反而更容易做对，召回率虚高，测不出索引的真实能力。

    从已有分布里取查询，得到的是"真实的用户会在库里找什么"，
    得到的召回率才有参考价值。
    """
    rng = random.Random(dataset.seed + 9973 if seed is None else seed)
    if n_queries < 1:
        raise ValueError(f"n_queries 必须 ≥ 1，收到 {n_queries}")

    queries: list[array] = []
    for _ in range(n_queries):
        source = dataset.vectors[rng.randrange(len(dataset.vectors))]
        jitter = array(
            "f", (rng.gauss(0.0, 1e-7) for _ in range(dataset.dim))
        )
        queries.append(array("f", (s + j for s, j in zip(source, jitter))))
    return queries
