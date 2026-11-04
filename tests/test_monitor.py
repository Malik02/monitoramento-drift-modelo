import numpy as np
from prometheus_client import CollectorRegistry, generate_latest

from monitoramento import config, simulador
from monitoramento.monitor import Monitor


def _monitor(clientes, modelo):
    ref = clientes.iloc[:2000]
    score = modelo.predict_proba(ref[config.FEATURES])[:, 1]
    return Monitor(ref, score, auc_baseline=0.8, registro=CollectorRegistry())


def test_monitor_exporta_metricas(clientes, modelo_sintetico):
    monitor = _monitor(clientes, modelo_sintetico)
    lote = clientes.iloc[2000:2500][config.FEATURES]
    scores = modelo_sintetico.predict_proba(lote)[:, 1]
    r = monitor.avaliar_lote(1, lote, scores)
    assert max(r["psi"].values()) < 0.1
    monitor.avaliar_rotulos(1, clientes.iloc[2000:2500][config.ALVO].to_numpy(), scores)
    texto = generate_latest(monitor.registro).decode()
    assert 'monitor_psi{feature="LIMIT_BAL"}' in texto
    assert 'monitor_performance{metrica="auc"}' in texto
    assert "monitor_auc_baseline 0.8" in texto


def test_simulacao_curta_mostra_os_tres_incidentes(clientes, modelo_sintetico):
    monitor = _monitor(clientes, modelo_sintetico)
    linhas = simulador.executar(
        lambda lote: modelo_sintetico.predict_proba(lote)[:, 1],
        monitor,
        clientes.iloc[2000:],
        dias=50,
        segundos_por_dia=0,
        tamanho_lote=400,
        atraso=5,
    )
    por_dia = {linha["dia"]: linha for linha in linhas}
    assert por_dia[5]["psi_LIMIT_BAL"] < config.PSI_ATENCAO
    assert por_dia[25]["psi_LIMIT_BAL"] > config.PSI_CRITICO
    assert por_dia[25]["max_taxa_nulos"] == 0
    assert por_dia[29]["max_taxa_nulos"] > 0.2
    assert por_dia[29]["max_taxa_fora_intervalo"] > 0.05
    # dia 50 mede a performance do lote do dia 45 (já com drift de conceito)
    auc_normal = np.mean([por_dia[d]["auc"] for d in range(10, 20)])
    auc_conceito = np.mean([por_dia[d]["auc"] for d in range(46, 51)])
    assert auc_conceito < auc_normal - 0.05
    assert "dia_rotulado" not in por_dia[5]
