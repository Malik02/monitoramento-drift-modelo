"""Simulador de produção: envia lotes "diários" ao serviço e alimenta o monitor.

Uso:
    python -m monitoramento.simulador            # contra a API (URL_API), em tempo comprimido
    python -m monitoramento.simulador --offline  # sem API nem espera, pontua localmente

Em produção, o monitor leria as requisições de um log/feature store e os rótulos de
um data warehouse; aqui o simulador entrega os dois diretamente, mantendo o mesmo
atraso de rótulo que existiria na vida real.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from prometheus_client import start_http_server

from monitoramento import config, dados, incidentes, treino
from monitoramento.monitor import Monitor

log = logging.getLogger("simulador")

Pontuador = Callable[[pd.DataFrame], np.ndarray]


def _para_json(lote: pd.DataFrame) -> list[dict]:
    """Converte o lote em registros JSON (NaN vira null)."""
    registros = lote.to_dict(orient="records")
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
        for r in registros
    ]


class PontuadorAPI:
    """Envia o lote à API em requisições de `REGISTROS_POR_REQUISICAO` clientes."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.sessao = requests.Session()

    def aguardar(self, tentativas: int = 60) -> None:
        for i in range(tentativas):
            try:
                if self.sessao.get(f"{self.url}/health", timeout=2).ok:
                    log.info("API disponível em %s", self.url)
                    return
            except requests.RequestException:
                pass
            log.info("Aguardando a API subir (%d/%d)...", i + 1, tentativas)
            time.sleep(2)
        raise RuntimeError(f"API indisponível em {self.url}")

    def __call__(self, lote: pd.DataFrame) -> np.ndarray:
        scores: list[float] = []
        passo = config.REGISTROS_POR_REQUISICAO
        for inicio in range(0, len(lote), passo):
            pedaco = lote.iloc[inicio : inicio + passo]
            resposta = self.sessao.post(
                f"{self.url}/predict", json={"registros": _para_json(pedaco)}, timeout=10
            )
            resposta.raise_for_status()
            scores.extend(resposta.json()["scores"])
        return np.asarray(scores)

    def enviar_requisicoes_invalidas(self, n: int) -> None:
        """Simula um cliente upstream quebrado: payloads sem campos obrigatórios (HTTP 422)."""
        for _ in range(n):
            self.sessao.post(
                f"{self.url}/predict", json={"registros": [{"LIMIT_BAL": "abc"}]}, timeout=10
            )


def descrever_incidentes(dia: int) -> str:
    eventos = []
    k = incidentes.intensidade_drift_covariavel(dia)
    if k > 0:
        eventos.append(f"drift de covariável ({k:.0%})")
    if incidentes.em_incidente_qualidade(dia):
        eventos.append("INCIDENTE DE QUALIDADE")
    if incidentes.em_drift_conceito(dia):
        eventos.append("drift de conceito")
    return ", ".join(eventos) or "normal"


def executar(
    pontuar: Pontuador,
    monitor: Monitor,
    pool: pd.DataFrame,
    dias: int = config.DIAS_SIMULADOS,
    segundos_por_dia: float = config.SEGUNDOS_POR_DIA,
    tamanho_lote: int = config.TAMANHO_LOTE,
    atraso: int = config.ATRASO_ROTULO_DIAS,
    semente: int = config.SEMENTE,
    ao_incidente_qualidade: Callable[[], None] | None = None,
) -> list[dict]:
    """Roda a linha do tempo completa e devolve uma linha de resultado por dia."""
    rng = np.random.default_rng(semente)
    pendentes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    linha_do_tempo: list[dict] = []

    for dia in range(1, dias + 1):
        inicio = time.monotonic()
        lote, rotulos = incidentes.gerar_dia(pool, dia, tamanho_lote, rng)
        scores = pontuar(lote)
        if ao_incidente_qualidade and incidentes.em_incidente_qualidade(dia):
            ao_incidente_qualidade()

        r = monitor.avaliar_lote(dia, lote, scores)
        pendentes[dia] = (rotulos, scores)
        linha = {
            "dia": dia,
            "situacao": descrever_incidentes(dia),
            "psi_LIMIT_BAL": round(r["psi"]["LIMIT_BAL"], 4),
            "psi_AGE": round(r["psi"]["AGE"], 4),
            "psi_max": round(max(r["psi"].values()), 4),
            "psi_score": round(r["psi_score"], 4),
            "max_taxa_nulos": round(max(r["nulos"].values()), 4),
            "max_taxa_fora_intervalo": round(max(r["fora_intervalo"].values()), 4),
        }

        # Rótulos chegam com atraso: hoje sabemos quem inadimpliu no lote de (dia - atraso).
        dia_rotulado = dia - atraso
        if dia_rotulado in pendentes:
            y, s = pendentes.pop(dia_rotulado)
            perf = monitor.avaliar_rotulos(dia_rotulado, y, s)
            linha.update(
                dia_rotulado=dia_rotulado,
                auc=round(perf["auc"], 4),
                gini=round(perf["gini"], 4),
                ks=round(perf["ks"], 4),
            )
        linha_do_tempo.append(linha)

        log.info(
            "Dia %02d [%s] PSI máx=%.3f (LIMIT_BAL=%.3f AGE=%.3f) nulos máx=%.1f%% "
            "fora do intervalo máx=%.1f%% | AUC(dia %s)=%s",
            dia,
            linha["situacao"],
            linha["psi_max"],
            linha["psi_LIMIT_BAL"],
            linha["psi_AGE"],
            100 * linha["max_taxa_nulos"],
            100 * linha["max_taxa_fora_intervalo"],
            linha.get("dia_rotulado", "-"),
            f"{linha['auc']:.3f}" if "auc" in linha else "aguardando rótulos",
        )
        espera = segundos_por_dia - (time.monotonic() - inicio)
        if espera > 0:
            time.sleep(espera)
    return linha_do_tempo


def criar_monitor(modelo, registro=None) -> Monitor:
    referencia = dados.carregar("referencia")
    score_ref = modelo.predict_proba(referencia[config.FEATURES])[:, 1]
    auc_base = treino.carregar_metadados()["holdout"]["auc"]
    return Monitor(referencia, score_ref, auc_base, registro=registro)


def salvar_csv(linhas: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(k for linha in linhas for k in linha))
    with caminho.open("w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    log.info("Linha do tempo salva em %s", caminho)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador de produção do modelo de crédito")
    parser.add_argument("--offline", action="store_true", help="pontua localmente, sem API")
    parser.add_argument("--saida", type=Path, default=os.environ.get("SAIDA_LINHA_DO_TEMPO"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    modelo = treino.carregar_modelo()
    pool = dados.carregar("producao")

    if args.offline:
        monitor = criar_monitor(modelo)
        linhas = executar(
            lambda lote: modelo.predict_proba(lote)[:, 1], monitor, pool, segundos_por_dia=0
        )
    else:
        monitor = criar_monitor(modelo)
        start_http_server(config.PORTA_MONITOR, registry=monitor.registro)
        log.info("Métricas do monitor em :%d/metrics", config.PORTA_MONITOR)
        api = PontuadorAPI(config.URL_API)
        api.aguardar()
        log.info(
            "Iniciando simulação: %d dias, %.1f s por dia, %d clientes por dia, "
            "rótulos com %d dias de atraso",
            config.DIAS_SIMULADOS,
            config.SEGUNDOS_POR_DIA,
            config.TAMANHO_LOTE,
            config.ATRASO_ROTULO_DIAS,
        )
        linhas = executar(
            api, monitor, pool, ao_incidente_qualidade=lambda: api.enviar_requisicoes_invalidas(3)
        )

    if args.saida:
        salvar_csv(linhas, args.saida)

    if not args.offline:
        log.info("Simulação concluída. Mantendo as métricas expostas (Ctrl+C para sair).")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
