"""数据集与索引的会话管理。

为什么需要一个 registry 而不是让 HTTP 层自己 new 一个对象：
索引**有状态且有成本**——构建一次要几百毫秒到几十秒，
而且每个索引都占着内存（ codes 或整张图）。HTTP 是无状态的，
每次请求重建索引会让"同一个索引反复付款"，
而不回收又会让内存无限增长。

这里做的是：以「数据集 + 引擎 + 参数」为键缓存索引，用 LRU 回收。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from .benchmark import BenchmarkResult, BenchmarkSpec, run_benchmark
from .dataset import Dataset, generate
from .hnsw import HNSWConfig, HNSWIndex
from .ivfpq import IVFPQConfig, IVFPQIndex
from .metrics import METRIC_INNER_PRODUCT, Metric, VectorStore, make_metric

#: 同时保留的数据集数量上限。
#: 每个 1000×32 的数据集连索引约 300KB，20 份就是 6MB ——
#: 这个上限不是为了防 OOM，而是为了让"忘了一直在发请求"这件事
#: 有一个明确的终点，而不是等到内存告警。
MAX_DATASETS = 20


@dataclass
class DatasetRecord:
    id: str
    dataset: Dataset
    store: VectorStore
    metric: Metric
    created_at: float = field(default_factory=time.time)
    #: 已构建的索引，键是引擎名
    indexes: dict[str, object] = field(default_factory=dict)
    #: 键的历史顺序（用于 LRU 回收数据集）
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()

    def describe(self) -> dict[str, object]:
        return {
            "id": self.id,
            "dataset": self.dataset.describe(),
            "store_revision": self.store.revision,
            "vector_bytes": self.store.memory_bytes(),
            "stats": self.store.stats(),
            "built_indexes": sorted(self.indexes),
            "created_seconds_ago": round(time.time() - self.created_at, 2),
        }


class DatasetNotFoundError(KeyError):
    pass


class IndexRegistry:
    """线程安全的注册表。

    用锁是因为 uvicorn 跑多线程时"两个请求同时建同一个数据集"
    会让后者覆盖前者，而两个请求都拿着自己的索引 ——
    表面正常，实际索引已经不属于任何一个请求的预期。
    """

    def __init__(self, max_datasets: int = MAX_DATASETS) -> None:
        self.max_datasets = max_datasets
        self._records: dict[str, DatasetRecord] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    # -- 数据集 --------------------------------------------------------

    def create_dataset(
        self,
        *,
        count: int,
        dim: int,
        kind: str,
        metric: str,
        seed: int,
        n_clusters: int,
        overlap: float,
    ) -> DatasetRecord:
        dataset = generate(
            count=count,
            dim=dim,
            kind=kind,
            metric=metric,
            seed=seed,
            n_clusters=n_clusters,
            overlap=overlap,
        )
        metric_obj = make_metric(metric)
        store = VectorStore(dim, metric_obj)
        for index, vector in enumerate(dataset.vectors):
            store.add(vector, label=dataset.labels[index] if dataset.labels else "")

        record = DatasetRecord(
            id=uuid.uuid4().hex[:12],
            dataset=dataset,
            store=store,
            metric=metric_obj,
        )
        with self._lock:
            self._records[record.id] = record
            self._order.append(record.id)
            self._evict_locked()
        return record

    def get(self, dataset_id: str) -> DatasetRecord:
        with self._lock:
            record = self._records.get(dataset_id)
            if record is None:
                raise DatasetNotFoundError(f"没有 id 为 {dataset_id!r} 的数据集")
            record.touch()
            return record

    def drop(self, dataset_id: str) -> bool:
        with self._lock:
            record = self._records.pop(dataset_id, None)
            if record is None:
                return False
            self._order = [k for k in self._order if k != dataset_id]
            return True

    def list_all(self) -> list[dict[str, object]]:
        with self._lock:
            return [r.describe() for r in self._records.values()]

    def _evict_locked(self) -> None:
        while len(self._order) > self.max_datasets:
            oldest = min(self._order, key=lambda k: self._records[k].last_used)
            self._records.pop(oldest, None)
            self._order.remove(oldest)

    # -- 索引 ----------------------------------------------------------

    def build_index(
        self, dataset_id: str, engine: str, params: dict[str, object]
    ) -> dict[str, object]:
        """按参数构建一个索引，缓存在 record 上。"""
        record = self.get(dataset_id)
        store = record.store
        metric = record.metric

        if engine == "hnsw":
            config = HNSWConfig(
                m=int(params.get("m", 16)),
                ef_construction=int(params.get("ef_construction", 100)),
                ef_search=int(params.get("ef_search", 64)),
                seed=int(params.get("seed", record.dataset.seed)),
            )
            index = HNSWIndex(store, metric, config)
            index.build()
        elif engine == "ivfpq":
            if metric.name == METRIC_INNER_PRODUCT:
                raise ValueError(
                    "IVF-PQ 不支持内积度量，请改用 cosine 或 l2"
                )
            config = IVFPQConfig(
                nlist=int(params.get("nlist", 16)),
                m=int(params.get("m", 8)),
                nbits=int(params.get("nbits", 4)),
                nprobe=int(params.get("nprobe", 4)),
                seed=int(params.get("seed", record.dataset.seed)),
            )
            index = IVFPQIndex(store, metric, config)
            index.build()
        else:
            raise ValueError(f"未知的引擎 {engine!r}")

        # 索引建完之后立刻置换记录上的旧索引：
        # 两个请求先后 build 同一个引擎时，后者胜出是预期行为
        # （后者的参数才是用户想要的），这里不做版本判断。
        record.indexes[engine] = index
        return index.snapshot()

    def get_index(self, dataset_id: str, engine: str):
        record = self.get(dataset_id)
        index = record.indexes.get(engine)
        if index is None:
            raise DatasetNotFoundError(
                f"数据集 {dataset_id!r} 上还没有构建 {engine!r} 索引"
            )
        return index

    # -- 基准 ----------------------------------------------------------

    def run_benchmark(self, spec: BenchmarkSpec) -> BenchmarkResult:
        return run_benchmark(spec)
