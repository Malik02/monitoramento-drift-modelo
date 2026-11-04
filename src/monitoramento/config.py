"""Configurações centrais: caminhos, features, faixas válidas e linha do tempo da simulação."""

from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(os.environ.get("MONITORAMENTO_RAIZ", Path(__file__).resolve().parents[2]))
DIR_DADOS_BRUTOS = RAIZ / "dados" / "brutos"
DIR_DADOS_PROCESSADOS = RAIZ / "dados" / "processados"
DIR_ARTEFATOS = RAIZ / "artefatos"

CAMINHO_MODELO = DIR_ARTEFATOS / "modelo.joblib"
CAMINHO_METADADOS = DIR_ARTEFATOS / "metadados.json"

URL_DADOS = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

SEMENTE = 42
ALVO = "default"

# Features do dataset "Default of Credit Card Clients" (UCI, Taiwan, 2005).
FEATURES_CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE"]
FEATURES_ATRASO = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
FEATURES_FATURA = [f"BILL_AMT{i}" for i in range(1, 7)]
FEATURES_PAGAMENTO = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = (
    ["LIMIT_BAL", "AGE"]
    + FEATURES_CATEGORICAS
    + FEATURES_ATRASO
    + FEATURES_FATURA
    + FEATURES_PAGAMENTO
)

# Faixas de domínio (regras de negócio). Valores fora delas são "fora do intervalo",
# mesmo que o modelo consiga pontuá-los: é exatamente o tipo de erro silencioso
# que queremos pegar antes que ele vire decisão de crédito errada.
FAIXAS_VALIDAS: dict[str, tuple[float, float]] = {
    "LIMIT_BAL": (1_000, 2_000_000),
    "AGE": (18, 100),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    **{f: (-2, 9) for f in FEATURES_ATRASO},
    **{f: (-1_000_000, 2_000_000) for f in FEATURES_FATURA},
    **{f: (0, 2_000_000) for f in FEATURES_PAGAMENTO},
}

# Limiares de monitoramento (convenção de mercado para PSI: 0,1 / 0,2).
PSI_ATENCAO = 0.1
PSI_CRITICO = 0.2
LIMIAR_TAXA_NULOS = 0.05
LIMIAR_FORA_INTERVALO = 0.01


def _int_env(nome: str, padrao: int) -> int:
    return int(os.environ.get(nome, padrao))


def _float_env(nome: str, padrao: float) -> float:
    return float(os.environ.get(nome, padrao))


# Linha do tempo da simulação (dias simulados). Comprimida para caber numa demo.
DIAS_SIMULADOS = _int_env("DIAS_SIMULADOS", 60)
SEGUNDOS_POR_DIA = _float_env("SEGUNDOS_POR_DIA", 4.0)
TAMANHO_LOTE = _int_env("TAMANHO_LOTE", 600)
REGISTROS_POR_REQUISICAO = _int_env("REGISTROS_POR_REQUISICAO", 50)
ATRASO_ROTULO_DIAS = _int_env("ATRASO_ROTULO_DIAS", 5)

DIA_INICIO_DRIFT_COVARIAVEL = 15  # campanha nova: limites maiores e clientes mais jovens
DIAS_RAMPA_DRIFT = 5  # o drift entra gradualmente ao longo de 5 dias
DIA_INICIO_INCIDENTE_QUALIDADE = 28  # pipeline upstream quebra: nulos e valores inválidos
DIA_FIM_INCIDENTE_QUALIDADE = 30  # inclusivo
DIA_INICIO_DRIFT_CONCEITO = 40  # choque econômico: a relação X -> inadimplência muda

URL_API = os.environ.get("URL_API", "http://localhost:8000")
PORTA_MONITOR = _int_env("PORTA_MONITOR", 8001)
