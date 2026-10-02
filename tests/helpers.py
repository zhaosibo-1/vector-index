"""测试共用的构造辅助函数。

放在这里而不是 ``conftest.py`` 的 fixture 里，是因为一部分测试需要
**现场搭一个 store**（比如要指定 metric、要预设 revision），
另一部分只需要拿一堆向量。用函数比用 fixture 更直接，
也让每个测试文件的自包含性更强 —— 读测试时不用跳去 conftest。
"""

from __future__ import annotations

from array import array
from typing import Iterable

from app.metrics import Metric, VectorStore, make_metric

#: 小规模数据集用的维度。取 8 是为了让手写期望值还算得动：
#: 32 维下"期望距离"没人能心算，测试里的断言就会退化成抄实现输出。
SMALL_DIM = 8


def make_store(dim: int = SMALL_DIM, metric: str = "l2") -> VectorStore:
    return VectorStore(dim, make_metric(metric))


def add_all(
    store: VectorStore, vectors: Iterable[Iterable[float]], labels: bool = False
) -> list[int]:
    """批量入库，返回 key 列表。"""
    keys: list[int] = []
    for i, v in enumerate(vectors):
        keys.append(store.add(v, label=f"v{i}" if labels else ""))
    return keys


def grid_vectors(store: VectorStore, side: int = 3) -> list[array]:
    """在 [0, 1]^dim 里铺一个 side^dim 的均匀网格。

    用网格而不是随机点，是因为测试要对**具体的距离关系**下断言
    （"A 比 B 近"）。随机点在 seed 固定时也可复现，
    但换 seed 一跑就红，维护成本高；网格是确定的，
    而且点与点的距离谱系清清楚楚。
    """
    dim = store.dim
    cells = side**dim
    out: list[array] = []
    for idx in range(cells):
        coords: list[float] = []
        rest = idx
        for _ in range(dim):
            coords.append(rest % side / (side - 1))
            rest //= side
        out.append(array("f", coords))
    return out


def as_array(values: Iterable[float]) -> array:
    return array("f", values)
