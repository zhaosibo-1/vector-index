"""HTTP 层：把底层索引暴露成可以点的东西。

只有这一个模块引入第三方依赖（FastAPI）。
算法层（metrics / dataset / bruteforce / hnsw / ivfpq / benchmark / registry）
全部零第三方依赖 —— 由 scripts/check_core_isolation.py 在 CI 里强制，
不是写在 README 里的承诺。
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STARTED_AT = time.time()

from . import __version__
from .benchmark import BenchmarkSpec
from .bruteforce import exact_top_k, recall_at_k, threshold_recall_at_k
from .dataset import MAX_COUNT, MAX_DIM, MIN_COUNT, MIN_DIM, VALID_KINDS
from .metrics import VALID_METRICS
from .registry import DatasetNotFoundError, IndexRegistry

app = FastAPI(
    title="vector-index",
    version=__version__,
    description="纯 Python 实现的 HNSW 与 IVF-PQ 向量索引，带可复核的召回率评测",
)

# 只在本地开发放行所有来源，生产环境应当收敛到具体域名。
# 这里 allow_credentials 保持 False 是刻意的：
# 一旦同时允许任意来源和携带凭证，跨域调用就能直接借用用户的登录态，
# 而 STAR 的浏览器行为又会让人误以为"配上 CORS 头就安全了"。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

registry = IndexRegistry()

WEB_DIR = "web"


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class DatasetRequest(BaseModel):
    count: int = Field(default=1000, ge=MIN_COUNT, le=MAX_COUNT)
    dim: int = Field(default=32, ge=MIN_DIM, le=MAX_DIM)
    kind: str = Field(default="clustered")
    metric: str = Field(default="l2")
    seed: int = Field(default=42)
    n_clusters: int = Field(default=24, ge=1)
    overlap: float = Field(default=0.15, gt=0)


class IndexRequest(BaseModel):
    engine: str = Field(default="hnsw")
    params: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    dataset_id: str
    engine: str = "hnsw"
    k: int = Field(default=10, ge=1, le=100)
    #: HNSW 专属：临时覆盖 ef_search
    ef_search: int | None = None
    #: IVF-PQ 专属：临时覆盖 nprobe
    nprobe: int | None = None
    #: 用库里的第几个向量当查询（默认第 0 个）
    probe_index: int = Field(default=0, ge=0)


class BenchmarkRequest(BaseModel):
    count: int = Field(default=1000, ge=MIN_COUNT, le=MAX_COUNT)
    dim: int = Field(default=32, ge=MIN_DIM, le=MAX_DIM)
    kind: str = "clustered"
    metric: str = "l2"
    seed: int = 42
    n_clusters: int = 24
    overlap: float = 0.15
    k: int = Field(default=10, ge=1, le=100)
    n_queries: int = Field(default=30, ge=1, le=500)
    engines: list[str] = Field(default_factory=lambda: ["brute", "hnsw", "ivfpq"])
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 64
    ivfpq_nlist: int = 16
    ivfpq_m: int = 8
    ivfpq_nbits: int = 4
    ivfpq_nprobe: int = 4
    sweep: bool = True


# ---------------------------------------------------------------------------
# 健康检查与元信息
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "dataset_count": len(registry.list_all()),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@app.get("/api/options")
def options() -> dict[str, object]:
    """前端下拉框要用的合法取值。

    放在服务端而不是写死在 JS 里，是为了避免"后端已经加了新 metric，
    前端还在用旧列表"这种不同步。
    """
    return {
        "metrics": list(VALID_METRICS),
        "kinds": list(VALID_KINDS),
        "engines": ["brute", "hnsw", "ivfpq"],
        "bounds": {
            "count": [MIN_COUNT, MAX_COUNT],
            "dim": [MIN_DIM, MAX_DIM],
        },
    }


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------


@app.post("/api/dataset")
def create_dataset(body: DatasetRequest) -> dict[str, object]:
    try:
        record = registry.create_dataset(**body.model_dump())
    except ValueError as exc:
        # 参数越界 → 422（客户端错误），而不是 500。
        # 500 会让调用方以为服务端炸了，进而重试 —— 而重试永远不会成功。
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record.describe()


@app.get("/api/dataset")
def list_datasets() -> dict[str, object]:
    return {"items": registry.list_all(), "count": len(registry.list_all())}


@app.get("/api/dataset/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, object]:
    try:
        return registry.get(dataset_id).describe()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/dataset/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, object]:
    return {"deleted": registry.drop(dataset_id)}


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------


@app.post("/api/index/{dataset_id}")
def build_index(dataset_id: str, body: IndexRequest) -> dict[str, object]:
    started = time.perf_counter()
    try:
        snapshot = registry.build_index(dataset_id, body.engine, body.params)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "dataset_id": dataset_id,
        "engine": body.engine,
        "build_wall_seconds": round(time.perf_counter() - started, 3),
        "snapshot": snapshot,
    }


@app.get("/api/index/{dataset_id}/{engine}")
def index_snapshot(dataset_id: str, engine: str) -> dict[str, object]:
    try:
        index = registry.get_index(dataset_id, engine)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "dataset_id": dataset_id,
        "engine": engine,
        "snapshot": index.snapshot(),
    }


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


@app.post("/api/search")
def search(body: SearchRequest) -> dict[str, object]:
    """用指定索引查一次，并与精确结果对比。

    返回**两种召回率**。这一点不能省：IVF-PQ 在数据有大量并列时，
    两种口径会差出 30 个点，只给一个数字会让使用者得出错误结论。
    """
    try:
        record = registry.get(body.dataset_id)
        index = registry.get_index(body.dataset_id, body.engine)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store = record.store
    metric = record.metric

    keys = store.all_keys()
    if not keys:
        raise HTTPException(status_code=422, detail="数据集是空的")
    probe_key = keys[body.probe_index % len(keys)]

    query = store.prepare_query(store.values_of(probe_key))
    extra: dict[str, int] = {}
    if body.engine == "hnsw" and body.ef_search:
        extra["ef_search"] = body.ef_search
    if body.engine == "ivfpq" and body.nprobe:
        extra["nprobe"] = body.nprobe

    ann_started = time.perf_counter()
    ann_hits = index.search(query[0], query[1], k=body.k, **extra)
    ann_ms = (time.perf_counter() - ann_started) * 1000.0

    exact_started = time.perf_counter()
    exact = exact_top_k(store, query, metric, k=body.k)
    exact_ms = (time.perf_counter() - exact_started) * 1000.0

    ann_keys = [key for _, key in ann_hits]
    return {
        "dataset_id": body.dataset_id,
        "engine": body.engine,
        "probe_key": probe_key,
        "probe_label": store.label_of(probe_key),
        "k": body.k,
        "ann": {
            "keys": ann_keys,
            "labels": [store.label_of(k) for k in ann_keys],
            "distances": [round(d, 6) for d, _ in ann_hits],
        },
        "exact": {
            "keys": exact.keys,
            "labels": [store.label_of(k) for k in exact.keys],
            "distances": [round(d, 6) for d in exact.distances],
            "threshold": (
                None if exact.threshold == float("inf")
                else round(exact.threshold, 6)
            ),
        },
        "recall": round(recall_at_k(ann_keys, exact), 4),
        "threshold_recall": round(threshold_recall_at_k(ann_hits, exact), 4),
        "latency": {
            "ann_ms": round(ann_ms, 4),
            "exact_ms": round(exact_ms, 4),
            "speedup": round(exact_ms / ann_ms, 2) if ann_ms > 0 else None,
        },
        "overrides": extra,
    }


# ---------------------------------------------------------------------------
# 基准
# ---------------------------------------------------------------------------


@app.post("/api/benchmark")
def benchmark(body: BenchmarkRequest) -> dict[str, object]:
    spec = BenchmarkSpec(
        count=body.count,
        dim=body.dim,
        kind=body.kind,
        metric=body.metric,
        seed=body.seed,
        n_clusters=body.n_clusters,
        overlap=body.overlap,
        k=body.k,
        n_queries=body.n_queries,
        engines=tuple(body.engines),
        hnsw_m=body.hnsw_m,
        hnsw_ef_construction=body.hnsw_ef_construction,
        hnsw_ef_search=body.hnsw_ef_search,
        ivfpq_nlist=body.ivfpq_nlist,
        ivfpq_m=body.ivfpq_m,
        ivfpq_nbits=body.ivfpq_nbits,
        ivfpq_nprobe=body.ivfpq_nprobe,
        sweep=body.sweep,
    )
    try:
        result = registry.run_benchmark(spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"spec": spec.describe(), "result": result.to_wire()}


@app.get("/api/benchmark")
def benchmark_get(
    count: int = Query(default=1000, ge=MIN_COUNT, le=MAX_COUNT),
    dim: int = Query(default=32, ge=MIN_DIM, le=MAX_DIM),
    metric: str = Query(default="l2"),
    seed: int = Query(default=42),
    k: int = Query(default=10, ge=1, le=100),
    n_queries: int = Query(default=30, ge=1, le=500),
    sweep: bool = Query(default=True),
) -> dict[str, object]:
    """GET 版本：方便在浏览器地址栏里直接跑一次。"""
    body = BenchmarkRequest(
        count=count, dim=dim, metric=metric, seed=seed,
        k=k, n_queries=n_queries, sweep=sweep,
    )
    return benchmark(body)


# ---------------------------------------------------------------------------
# 前端
# ---------------------------------------------------------------------------


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(f"{WEB_DIR}/index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
