"""Download e preparação do dataset público "Default of Credit Card Clients" (UCI)."""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from monitoramento import config

log = logging.getLogger(__name__)

ARQUIVO_XLS = "default of credit card clients.xls"


def baixar_dados(forcar: bool = False) -> pd.DataFrame:
    """Baixa o zip da UCI (uma vez) e devolve o DataFrame limpo."""
    config.DIR_DADOS_BRUTOS.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_DADOS_BRUTOS / "credit_default.csv"
    if destino.exists() and not forcar:
        return pd.read_csv(destino)

    log.info("Baixando dataset da UCI: %s", config.URL_DADOS)
    resposta = requests.get(config.URL_DADOS, timeout=120)
    resposta.raise_for_status()
    with (
        zipfile.ZipFile(io.BytesIO(resposta.content)) as zf,
        zf.open(ARQUIVO_XLS) as f,
    ):
        df = pd.read_excel(f, header=1, engine="xlrd")

    df = df.rename(columns={"default payment next month": config.ALVO}).drop(columns=["ID"])
    df = df[config.FEATURES + [config.ALVO]].astype(float)
    df[config.ALVO] = df[config.ALVO].astype(int)
    df.to_csv(destino, index=False)
    log.info("Dataset salvo em %s (%d linhas)", destino, len(df))
    return df


def dividir_dados(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Divide em treino (60%), referência/holdout (20%) e pool de produção (20%).

    - treino: ajusta o modelo;
    - referencia: holdout para a AUC de baseline e distribuição de referência do PSI;
    - producao: clientes "futuros" que o simulador sorteia a cada dia.
    """
    treino, resto = train_test_split(
        df, test_size=0.4, random_state=config.SEMENTE, stratify=df[config.ALVO]
    )
    referencia, producao = train_test_split(
        resto, test_size=0.5, random_state=config.SEMENTE, stratify=resto[config.ALVO]
    )
    return {
        "treino": treino.reset_index(drop=True),
        "referencia": referencia.reset_index(drop=True),
        "producao": producao.reset_index(drop=True),
    }


def preparar() -> dict[str, pd.DataFrame]:
    """Baixa, divide e salva os conjuntos em dados/processados."""
    partes = dividir_dados(baixar_dados())
    config.DIR_DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    for nome, parte in partes.items():
        parte.to_csv(config.DIR_DADOS_PROCESSADOS / f"{nome}.csv", index=False)
    return partes


def carregar(nome: str) -> pd.DataFrame:
    """Carrega um conjunto processado ('treino', 'referencia' ou 'producao')."""
    caminho = config.DIR_DADOS_PROCESSADOS / f"{nome}.csv"
    if not caminho.exists():
        preparar()
    return pd.read_csv(caminho)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for nome, parte in preparar().items():
        log.info("%s: %d linhas, taxa de default %.3f", nome, len(parte), parte[config.ALVO].mean())
