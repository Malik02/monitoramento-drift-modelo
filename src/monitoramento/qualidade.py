"""Checagens de qualidade dos dados de entrada: nulos e valores fora do intervalo de domínio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoramento import config


def taxa_nulos(lote: pd.DataFrame, features: list[str] | None = None) -> dict[str, float]:
    """Fração de valores nulos (ou ausentes) por feature."""
    features = features or config.FEATURES
    n = len(lote)
    if n == 0:
        return {f: float("nan") for f in features}
    return {f: float(lote[f].isna().mean()) if f in lote.columns else 1.0 for f in features}


def taxa_fora_intervalo(
    lote: pd.DataFrame,
    faixas: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Fração de valores **não nulos** fora da faixa válida de domínio, por feature.

    O denominador é o total de linhas, para que nulos e fora-do-intervalo sejam
    comparáveis e somáveis no painel.
    """
    faixas = faixas or config.FAIXAS_VALIDAS
    n = len(lote)
    resultado: dict[str, float] = {}
    for f, (minimo, maximo) in faixas.items():
        if n == 0 or f not in lote.columns:
            resultado[f] = float("nan")
            continue
        v = lote[f].to_numpy(dtype=float)
        validos = ~np.isnan(v)
        fora = validos & ((v < minimo) | (v > maximo))
        resultado[f] = float(fora.sum() / n)
    return resultado


def resumo_qualidade(lote: pd.DataFrame) -> dict[str, float]:
    """Indicadores agregados do lote: pior taxa de nulos e pior taxa fora do intervalo."""
    nulos = taxa_nulos(lote)
    fora = taxa_fora_intervalo(lote)
    return {
        "max_taxa_nulos": max(nulos.values()),
        "max_taxa_fora_intervalo": max(fora.values()),
        "linhas_com_problema": float(
            (lote[config.FEATURES].isna().any(axis=1)).mean() if len(lote) else float("nan")
        ),
    }
