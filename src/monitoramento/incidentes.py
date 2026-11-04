"""Injeção de problemas realistas em pontos conhecidos da linha do tempo simulada.

Três tipos de incidente, cada um pego por um sinal diferente do monitor:

1. Drift de covariável (dia 15 em diante, em rampa de 5 dias): uma campanha de
   aquisição traz clientes com **limite maior** e **mais jovens**. A distribuição de
   X muda, a relação X -> y não. Sinal: PSI/KS de LIMIT_BAL e AGE.
2. Incidente de qualidade (dias 28 a 30): uma mudança no pipeline upstream gera
   **nulos** em PAY_0, BILL_AMT1 e PAY_AMT1, idade com sentinela -1 e EDUCATION
   com código inexistente. Sinal: taxa de nulos e de fora do intervalo.
3. Drift de conceito (dia 40 em diante): um choque econômico faz clientes até
   então adimplentes passarem a inadimplir, e parte dos atrasados se recupera.
   X não muda, mas a relação X -> y muda. Sinal: só a performance com rótulo
   atrasado (AUC/KS/Gini), que chega dias depois.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoramento import config


def intensidade_drift_covariavel(dia: int) -> float:
    """0 antes do início, sobe linearmente durante a rampa e fica em 1 depois."""
    progresso = (dia - config.DIA_INICIO_DRIFT_COVARIAVEL + 1) / config.DIAS_RAMPA_DRIFT
    return float(np.clip(progresso, 0.0, 1.0))


def em_incidente_qualidade(dia: int) -> bool:
    return config.DIA_INICIO_INCIDENTE_QUALIDADE <= dia <= config.DIA_FIM_INCIDENTE_QUALIDADE


def em_drift_conceito(dia: int) -> bool:
    return dia >= config.DIA_INICIO_DRIFT_CONCEITO


def aplicar_drift_covariavel(lote: pd.DataFrame, dia: int) -> pd.DataFrame:
    """Limite até 50% maior e clientes até 5 anos mais jovens (mínimo 21)."""
    k = intensidade_drift_covariavel(dia)
    if k == 0:
        return lote
    lote = lote.copy()
    lote["LIMIT_BAL"] = (lote["LIMIT_BAL"] * (1 + 0.5 * k)).round(-3)
    lote["AGE"] = np.maximum(21, lote["AGE"] - np.round(5 * k))
    return lote


def aplicar_incidente_qualidade(
    lote: pd.DataFrame, dia: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Nulos (30%) em três colunas, AGE=-1 (10%) e EDUCATION=99 (5%)."""
    if not em_incidente_qualidade(dia):
        return lote
    lote = lote.copy()
    n = len(lote)
    for coluna in ("PAY_0", "BILL_AMT1", "PAY_AMT1"):
        lote.loc[rng.random(n) < 0.30, coluna] = np.nan
    lote.loc[rng.random(n) < 0.10, "AGE"] = -1
    lote.loc[rng.random(n) < 0.05, "EDUCATION"] = 99
    return lote


def aplicar_drift_conceito(
    lote: pd.DataFrame, rotulos: np.ndarray, dia: int, rng: np.random.Generator
) -> np.ndarray:
    """Altera os rótulos (o "mundo real"), não as features.

    - 12% dos clientes em dia (PAY_0 <= 0) que não inadimpliriam passam a inadimplir;
    - 35% dos atrasados (PAY_0 >= 1) que inadimpliriam acabam pagando.

    O modelo, treinado no regime antigo, continua confiando em PAY_0 e perde poder
    de separação.
    """
    if not em_drift_conceito(dia):
        return rotulos
    y = np.asarray(rotulos).copy()
    pay0 = lote["PAY_0"].to_numpy(dtype=float)
    em_dia = np.nan_to_num(pay0, nan=0) <= 0
    sorteio = rng.random(len(y))
    y[(y == 0) & em_dia & (sorteio < 0.12)] = 1
    y[(y == 1) & ~em_dia & (sorteio < 0.35)] = 0
    return y


def gerar_dia(
    pool: pd.DataFrame, dia: int, tamanho: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sorteia o lote do dia no pool de produção e aplica os incidentes ativos.

    Devolve (features, rótulos verdadeiros). Os rótulos só são revelados ao monitor
    `ATRASO_ROTULO_DIAS` depois, como acontece em crédito (a inadimplência leva
    tempo para ser observada).
    """
    amostra = pool.sample(n=tamanho, replace=True, random_state=rng.integers(1 << 31))
    amostra = amostra.reset_index(drop=True)
    rotulos = amostra[config.ALVO].to_numpy()
    features = amostra[config.FEATURES]
    # O drift de conceito usa as features "verdadeiras", antes do incidente de qualidade.
    rotulos = aplicar_drift_conceito(features, rotulos, dia, rng)
    features = aplicar_drift_covariavel(features, dia)
    features = aplicar_incidente_qualidade(features, dia, rng)
    return features, rotulos
