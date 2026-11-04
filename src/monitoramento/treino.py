"""Treino do modelo de risco de crédito (probabilidade de inadimplência no próximo mês)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from monitoramento import config, dados
from monitoramento.estatisticas import metricas_performance

log = logging.getLogger(__name__)


def criar_modelo() -> HistGradientBoostingClassifier:
    """Gradient boosting por histogramas.

    Escolhido porque lida nativamente com valores nulos (o serviço não quebra num
    incidente de qualidade; o monitor é que precisa avisar), é rápido e tem
    boa performance em dados tabulares sem muito ajuste.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=config.SEMENTE,
    )


def treinar() -> dict:
    """Treina, avalia no holdout de referência e salva modelo + metadados."""
    partes = dados.preparar()
    treino, referencia = partes["treino"], partes["referencia"]

    modelo = criar_modelo()
    modelo.fit(treino[config.FEATURES], treino[config.ALVO])

    score_ref = modelo.predict_proba(referencia[config.FEATURES])[:, 1]
    perf = metricas_performance(referencia[config.ALVO].to_numpy(), score_ref)

    config.DIR_ARTEFATOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, config.CAMINHO_MODELO)
    metadados = {
        "modelo": "HistGradientBoostingClassifier",
        "versao": "1.0.0",
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "semente": config.SEMENTE,
        "n_treino": len(treino),
        "n_holdout": len(referencia),
        "iteracoes": int(modelo.n_iter_),
        "holdout": {k: round(v, 4) for k, v in perf.items()},
        "features": config.FEATURES,
    }
    config.CAMINHO_METADADOS.write_text(json.dumps(metadados, indent=2, ensure_ascii=False))
    log.info(
        "Modelo salvo em %s | holdout AUC=%.4f Gini=%.4f KS=%.4f",
        config.CAMINHO_MODELO,
        perf["auc"],
        perf["gini"],
        perf["ks"],
    )
    return metadados


def carregar_modelo() -> HistGradientBoostingClassifier:
    """Carrega o modelo salvo; treina se ainda não existir."""
    if not config.CAMINHO_MODELO.exists():
        treinar()
    return joblib.load(config.CAMINHO_MODELO)


def carregar_metadados() -> dict:
    if not config.CAMINHO_METADADOS.exists():
        treinar()
    return json.loads(config.CAMINHO_METADADOS.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    treinar()
