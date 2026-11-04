"""Monitor de drift, qualidade e performance, exportado como métricas Prometheus.

Para cada lote diário o monitor calcula, contra a distribuição de referência
(holdout do treino):

- PSI e KS por feature (drift de covariável);
- PSI do score do modelo (drift de predição);
- taxa de nulos e de valores fora do intervalo por feature (qualidade);

e, quando os rótulos de um lote antigo chegam (com atraso), AUC, Gini e KS de
separação daquele lote (performance).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge

from monitoramento import config, estatisticas, qualidade

log = logging.getLogger(__name__)


class Monitor:
    def __init__(
        self,
        referencia: pd.DataFrame,
        score_referencia: np.ndarray,
        auc_baseline: float,
        registro: CollectorRegistry | None = None,
    ) -> None:
        self.referencia = referencia[config.FEATURES]
        self.score_referencia = np.asarray(score_referencia, dtype=float)
        # As faixas do PSI são fixadas uma vez, na referência (não no lote atual).
        self.faixas = {
            f: estatisticas.calcular_faixas(self.referencia[f].to_numpy()) for f in config.FEATURES
        }
        self.faixas_score = estatisticas.calcular_faixas(self.score_referencia)
        self.registro = registro or CollectorRegistry()
        self._criar_metricas(auc_baseline)

    def _criar_metricas(self, auc_baseline: float) -> None:
        r = self.registro
        self.g_dia = Gauge("monitor_dia_simulado", "Dia simulado do último lote", registry=r)
        self.g_psi = Gauge("monitor_psi", "PSI da feature vs. referência", ["feature"], registry=r)
        self.g_ks = Gauge(
            "monitor_ks_estatistica",
            "Estatística D do teste KS vs. referência",
            ["feature"],
            registry=r,
        )
        self.g_ks_p = Gauge(
            "monitor_ks_pvalor", "p-valor do teste KS vs. referência", ["feature"], registry=r
        )
        self.g_nulos = Gauge(
            "monitor_taxa_nulos", "Fração de nulos no lote", ["feature"], registry=r
        )
        self.g_fora = Gauge(
            "monitor_taxa_fora_intervalo",
            "Fração de valores fora da faixa válida",
            ["feature"],
            registry=r,
        )
        self.g_psi_score = Gauge(
            "monitor_psi_score", "PSI do score do modelo vs. referência", registry=r
        )
        self.g_score_medio = Gauge("monitor_score_medio", "Score médio do lote", registry=r)
        self.g_perf = Gauge(
            "monitor_performance",
            "Performance com rótulo atrasado (auc, gini, ks)",
            ["metrica"],
            registry=r,
        )
        self.g_dia_rotulado = Gauge(
            "monitor_dia_rotulado", "Dia do lote cuja performance foi medida", registry=r
        )
        self.g_default = Gauge(
            "monitor_taxa_default_observada",
            "Taxa de inadimplência observada no lote rotulado",
            registry=r,
        )
        self.g_auc_base = Gauge(
            "monitor_auc_baseline", "AUC do modelo no holdout de validação", registry=r
        )
        self.g_auc_base.set(auc_baseline)
        self.c_lotes = Counter("monitor_lotes_processados", "Lotes diários processados", registry=r)
        self.c_rotulados = Counter(
            "monitor_lotes_rotulados", "Lotes com rótulo recebido", registry=r
        )

    def avaliar_lote(self, dia: int, lote: pd.DataFrame, scores: np.ndarray) -> dict:
        """Drift e qualidade de um lote recém-pontuado. Atualiza as métricas."""
        resultado: dict = {"dia": dia, "psi": {}, "ks": {}}
        nulos = qualidade.taxa_nulos(lote)
        fora = qualidade.taxa_fora_intervalo(lote)
        for f in config.FEATURES:
            valores = lote[f].to_numpy(dtype=float)
            valor_psi = estatisticas.psi(
                self.referencia[f].to_numpy(), valores, faixas=self.faixas[f]
            )
            d, p = estatisticas.teste_ks(self.referencia[f].to_numpy(), valores)
            resultado["psi"][f] = valor_psi
            resultado["ks"][f] = d
            self.g_psi.labels(feature=f).set(valor_psi)
            self.g_ks.labels(feature=f).set(d)
            self.g_ks_p.labels(feature=f).set(p)
            self.g_nulos.labels(feature=f).set(nulos[f])
            self.g_fora.labels(feature=f).set(fora[f])

        scores = np.asarray(scores, dtype=float)
        resultado["psi_score"] = estatisticas.psi(
            self.score_referencia, scores, faixas=self.faixas_score
        )
        resultado["nulos"] = nulos
        resultado["fora_intervalo"] = fora
        self.g_psi_score.set(resultado["psi_score"])
        self.g_score_medio.set(float(np.mean(scores)))
        self.g_dia.set(dia)
        self.c_lotes.inc()
        return resultado

    def avaliar_rotulos(self, dia: int, rotulos: np.ndarray, scores: np.ndarray) -> dict:
        """Performance do lote do `dia`, cujos rótulos acabaram de chegar."""
        perf = estatisticas.metricas_performance(rotulos, scores)
        for nome, valor in perf.items():
            self.g_perf.labels(metrica=nome).set(valor)
        self.g_dia_rotulado.set(dia)
        self.g_default.set(float(np.mean(rotulos)))
        self.c_rotulados.inc()
        return perf
