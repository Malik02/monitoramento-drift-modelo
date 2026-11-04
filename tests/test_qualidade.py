import numpy as np
import pytest

from monitoramento import qualidade


def test_lote_limpo_nao_tem_problemas(clientes):
    nulos = qualidade.taxa_nulos(clientes)
    fora = qualidade.taxa_fora_intervalo(clientes)
    assert max(nulos.values()) == 0
    assert max(fora.values()) == 0


def test_taxa_nulos(clientes):
    lote = clientes.head(100).copy()
    lote.loc[:24, "PAY_0"] = np.nan  # 25 linhas
    assert qualidade.taxa_nulos(lote)["PAY_0"] == pytest.approx(0.25)
    assert qualidade.taxa_nulos(lote)["AGE"] == 0


def test_taxa_fora_intervalo_ignora_nulos(clientes):
    lote = clientes.head(100).copy()
    lote.loc[:9, "AGE"] = -1  # 10 fora do intervalo
    lote.loc[10:19, "AGE"] = np.nan  # nulo não conta como fora do intervalo
    lote.loc[20, "EDUCATION"] = 99
    fora = qualidade.taxa_fora_intervalo(lote)
    assert fora["AGE"] == pytest.approx(0.10)
    assert fora["EDUCATION"] == pytest.approx(0.01)


def test_coluna_ausente_conta_como_nula(clientes):
    lote = clientes.head(10).drop(columns=["AGE"])
    assert qualidade.taxa_nulos(lote)["AGE"] == 1.0


def test_resumo_qualidade(clientes):
    lote = clientes.head(200).copy()
    lote.loc[:19, "BILL_AMT1"] = np.nan
    resumo = qualidade.resumo_qualidade(lote)
    assert resumo["max_taxa_nulos"] == pytest.approx(0.10)
    assert resumo["linhas_com_problema"] == pytest.approx(0.10)
