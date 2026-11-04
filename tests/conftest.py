"""Fixtures compartilhadas: dados sintéticos com o mesmo esquema do dataset real.

Os testes não baixam nada da internet, então rodam no CI sem rede e em segundos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from monitoramento import config


def gerar_clientes(n: int, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.choice(np.arange(10_000, 500_001, 10_000), n).astype(float),
            "AGE": rng.integers(21, 70, n).astype(float),
            "SEX": rng.integers(1, 3, n).astype(float),
            "EDUCATION": rng.integers(1, 5, n).astype(float),
            "MARRIAGE": rng.integers(1, 4, n).astype(float),
        }
    )
    for f in config.FEATURES_ATRASO:
        df[f] = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[0.15, 0.2, 0.45, 0.1, 0.07, 0.03])
    for f in config.FEATURES_FATURA:
        df[f] = rng.gamma(1.5, 30_000, n).round()
    for f in config.FEATURES_PAGAMENTO:
        df[f] = rng.gamma(1.0, 4_000, n).round()
    logito = -1.6 + 0.9 * df["PAY_0"] - df["LIMIT_BAL"] / 400_000
    df[config.ALVO] = (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int)
    return df[config.FEATURES + [config.ALVO]].astype({f: float for f in config.FEATURES})


@pytest.fixture(scope="session")
def clientes() -> pd.DataFrame:
    return gerar_clientes(4000)


@pytest.fixture(scope="session")
def modelo_sintetico(clientes):
    modelo = HistGradientBoostingClassifier(max_iter=50, random_state=0)
    modelo.fit(clientes[config.FEATURES], clientes[config.ALVO])
    return modelo
