import numpy as np

from monitoramento import config, estatisticas, incidentes


def test_intensidade_da_rampa():
    inicio = config.DIA_INICIO_DRIFT_COVARIAVEL
    assert incidentes.intensidade_drift_covariavel(inicio - 1) == 0
    assert 0 < incidentes.intensidade_drift_covariavel(inicio) < 1
    assert incidentes.intensidade_drift_covariavel(inicio + config.DIAS_RAMPA_DRIFT) == 1


def test_dias_normais_nao_alteram_dados(clientes):
    rng = np.random.default_rng(0)
    lote = clientes[config.FEATURES].head(300)
    dia = 5
    assert incidentes.aplicar_drift_covariavel(lote, dia).equals(lote)
    assert incidentes.aplicar_incidente_qualidade(lote, dia, rng).equals(lote)
    y = clientes[config.ALVO].head(300).to_numpy()
    assert (incidentes.aplicar_drift_conceito(lote, y, dia, rng) == y).all()


def test_drift_covariavel_gera_psi_alto(clientes):
    lote = clientes[config.FEATURES]
    driftado = incidentes.aplicar_drift_covariavel(lote, 30)
    assert estatisticas.psi(lote["LIMIT_BAL"], driftado["LIMIT_BAL"]) > config.PSI_CRITICO
    assert estatisticas.psi(lote["AGE"], driftado["AGE"]) > config.PSI_CRITICO
    assert estatisticas.psi(lote["PAY_0"], driftado["PAY_0"]) == 0


def test_incidente_qualidade_gera_nulos_e_invalidos(clientes):
    rng = np.random.default_rng(1)
    lote = incidentes.aplicar_incidente_qualidade(
        clientes[config.FEATURES], config.DIA_INICIO_INCIDENTE_QUALIDADE, rng
    )
    assert 0.25 < lote["PAY_0"].isna().mean() < 0.35
    assert 0.05 < (lote["AGE"] == -1).mean() < 0.15
    assert (lote["EDUCATION"] == 99).any()


def test_drift_conceito_muda_rotulos_e_derruba_auc(clientes, modelo_sintetico):
    rng = np.random.default_rng(2)
    X = clientes[config.FEATURES]
    y = clientes[config.ALVO].to_numpy()
    y_novo = incidentes.aplicar_drift_conceito(X, y, config.DIA_INICIO_DRIFT_CONCEITO, rng)
    score = modelo_sintetico.predict_proba(X)[:, 1]
    auc_antes = estatisticas.metricas_performance(y, score)["auc"]
    auc_depois = estatisticas.metricas_performance(y_novo, score)["auc"]
    assert auc_depois < auc_antes - 0.05


def test_gerar_dia_e_reprodutivel(clientes):
    a, ya = incidentes.gerar_dia(clientes, 29, 100, np.random.default_rng(7))
    b, yb = incidentes.gerar_dia(clientes, 29, 100, np.random.default_rng(7))
    assert a.equals(b) and (ya == yb).all()
    assert list(a.columns) == config.FEATURES
