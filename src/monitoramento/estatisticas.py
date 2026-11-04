r"""Estatísticas de drift e de performance, implementadas à mão onde importa.

PSI (Population Stability Index)
--------------------------------
Dado um conjunto de faixas (bins) definido na distribuição de **referência**,
com proporções :math:`r_i` (referência) e :math:`a_i` (atual/produção):

.. math::

    PSI = \sum_i (a_i - r_i) \cdot \ln\left(\frac{a_i}{r_i}\right)

Cada termo é sempre >= 0 (se a_i > r_i, os dois fatores são positivos; se
a_i < r_i, os dois são negativos). O PSI é a divergência de KL simétrica
(divergência de Jeffreys) entre as duas distribuições discretizadas:
PSI = KL(a || r) + KL(r || a).

Convenção de mercado (crédito):
    PSI < 0,1        -> estável
    0,1 <= PSI < 0,2 -> atenção (mudança moderada)
    PSI >= 0,2       -> mudança significativa: investigar / considerar retreino
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

EPSILON = 1e-4  # evita log(0) e divisão por zero em faixas vazias
MAX_VALORES_DISCRETOS = 12  # até 12 valores distintos: cada valor vira uma faixa


def calcular_faixas(referencia: np.ndarray, n_faixas: int = 10) -> np.ndarray:
    """Define as bordas das faixas a partir da referência.

    - Variáveis discretas (poucos valores distintos, ex.: PAY_0, EDUCATION):
      uma faixa por valor, com bordas nos pontos médios.
    - Variáveis contínuas: quantis da referência (faixas de mesma população).

    As bordas extremas são -inf e +inf, então valores novos nunca "caem fora".
    """
    ref = np.asarray(referencia, dtype=float)
    ref = ref[~np.isnan(ref)]
    valores = np.unique(ref)
    if len(valores) <= MAX_VALORES_DISCRETOS:
        internas = (valores[:-1] + valores[1:]) / 2
    else:
        quantis = np.quantile(ref, np.linspace(0, 1, n_faixas + 1)[1:-1])
        internas = np.unique(quantis)
    return np.concatenate([[-np.inf], internas, [np.inf]])


def proporcoes(valores: np.ndarray, faixas: np.ndarray) -> np.ndarray:
    """Proporção de observações (não nulas) em cada faixa."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.full(len(faixas) - 1, np.nan)
    contagens, _ = np.histogram(v, bins=faixas)
    return contagens / len(v)


def psi_de_proporcoes(ref: np.ndarray, atual: np.ndarray) -> float:
    """PSI = soma (a_i - r_i) * ln(a_i / r_i), com piso EPSILON nas proporções."""
    r = np.clip(np.asarray(ref, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(atual, dtype=float), EPSILON, None)
    return float(np.sum((a - r) * np.log(a / r)))


def psi(
    referencia: np.ndarray,
    atual: np.ndarray,
    n_faixas: int = 10,
    faixas: np.ndarray | None = None,
) -> float:
    """PSI entre duas amostras. Nulos são ignorados (são medidos à parte, em qualidade)."""
    if faixas is None:
        faixas = calcular_faixas(referencia, n_faixas)
    prop_atual = proporcoes(atual, faixas)
    if np.isnan(prop_atual).all():
        return float("nan")
    return psi_de_proporcoes(proporcoes(referencia, faixas), prop_atual)


def teste_ks(referencia: np.ndarray, atual: np.ndarray) -> tuple[float, float]:
    """Teste de Kolmogorov-Smirnov de duas amostras: (estatística D, p-valor)."""
    r = np.asarray(referencia, dtype=float)
    a = np.asarray(atual, dtype=float)
    r, a = r[~np.isnan(r)], a[~np.isnan(a)]
    if len(r) == 0 or len(a) == 0:
        return float("nan"), float("nan")
    resultado = stats.ks_2samp(r, a)
    return float(resultado.statistic), float(resultado.pvalue)


def ks_separacao(y_true: np.ndarray, score: np.ndarray) -> float:
    """KS de separação do modelo: máxima distância entre as CDFs de score de bons e maus."""
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    maus, bons = s[y == 1], s[y == 0]
    if len(maus) == 0 or len(bons) == 0:
        return float("nan")
    return float(stats.ks_2samp(maus, bons).statistic)


def metricas_performance(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUC, Gini (= 2*AUC - 1) e KS de separação. NaN se houver só uma classe."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "gini": float("nan"), "ks": float("nan")}
    auc = float(roc_auc_score(y, score))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks_separacao(y, score)}
