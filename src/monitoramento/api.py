"""Serviço de escoragem (FastAPI) com métricas Prometheus em /metrics."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, create_model

from monitoramento import config, treino

log = logging.getLogger(__name__)

REGISTRO = CollectorRegistry()
REQUISICOES = Counter(
    "api_requisicoes", "Requisições HTTP recebidas", ["rota", "status"], registry=REGISTRO
)
ERROS = Counter(
    "api_erros", "Requisições com erro (status >= 400)", ["rota", "status"], registry=REGISTRO
)
LATENCIA = Histogram(
    "api_latencia_segundos",
    "Latência das requisições HTTP",
    ["rota"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRO,
)
SCORES = Histogram(
    "api_score_predicao",
    "Distribuição dos scores (probabilidade de inadimplência) servidos",
    buckets=tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2)),
    registry=REGISTRO,
)
REGISTROS_PREDITOS = Counter("api_registros_preditos", "Clientes pontuados", registry=REGISTRO)
NULOS_RECEBIDOS = Counter(
    "api_valores_nulos", "Valores nulos recebidos por feature", ["feature"], registry=REGISTRO
)
MODELO_INFO = Gauge(
    "api_modelo_info", "Versão do modelo carregado", ["versao", "modelo"], registry=REGISTRO
)

# Inicializa as séries de erro com zero: sem isso, o primeiro erro cria a série já
# com valor 1 e o rate() do Prometheus não enxerga esse incremento.
for _status in ("422", "500"):
    ERROS.labels(rota="/predict", status=_status)
    REQUISICOES.labels(rota="/predict", status=_status)

# Esquema de entrada: todas as features numéricas e opcionais (nulos são aceitos
# e pontuados, porque o modelo os trata; o monitor é quem acusa o problema).
Cliente = create_model(  # type: ignore[call-overload]
    "Cliente", **{f: (float | None, Field(...)) for f in config.FEATURES}
)


class PedidoPredicao(BaseModel):
    registros: list[Cliente] = Field(..., min_length=1, max_length=1000)  # type: ignore[valid-type]


class RespostaPredicao(BaseModel):
    scores: list[float]
    versao_modelo: str


ESTADO: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    ESTADO["modelo"] = treino.carregar_modelo()
    ESTADO["metadados"] = treino.carregar_metadados()
    MODELO_INFO.labels(
        versao=ESTADO["metadados"]["versao"], modelo=ESTADO["metadados"]["modelo"]
    ).set(1)
    log.info("Modelo carregado: %s", ESTADO["metadados"]["holdout"])
    yield
    ESTADO.clear()


app = FastAPI(title="Serviço de risco de crédito", version="0.1.0", lifespan=ciclo_de_vida)


@app.middleware("http")
async def medir(request: Request, call_next):
    rota = request.url.path
    inicio = time.perf_counter()
    try:
        resposta = await call_next(request)
        status = resposta.status_code
    except Exception:
        status = 500
        raise
    finally:
        if rota != "/metrics":
            LATENCIA.labels(rota=rota).observe(time.perf_counter() - inicio)
            REQUISICOES.labels(rota=rota, status=str(status)).inc()
            if status >= 400:
                ERROS.labels(rota=rota, status=str(status)).inc()
    return resposta


@app.get("/health")
def saude() -> dict:
    return {"status": "ok", "modelo_carregado": "modelo" in ESTADO}


@app.post("/predict", response_model=RespostaPredicao)
def prever(pedido: PedidoPredicao) -> RespostaPredicao:
    linhas = [r.model_dump() for r in pedido.registros]
    df = pd.DataFrame(linhas, columns=config.FEATURES).astype(float)
    for f, n in df.isna().sum().items():
        if n:
            NULOS_RECEBIDOS.labels(feature=f).inc(int(n))
    scores = ESTADO["modelo"].predict_proba(df)[:, 1]
    for s in scores:
        SCORES.observe(float(s))
    REGISTROS_PREDITOS.inc(len(scores))
    return RespostaPredicao(
        scores=[round(float(s), 6) for s in scores],
        versao_modelo=ESTADO["metadados"]["versao"],
    )


@app.get("/metrics")
def metricas() -> Response:
    return Response(generate_latest(REGISTRO), media_type=CONTENT_TYPE_LATEST)
