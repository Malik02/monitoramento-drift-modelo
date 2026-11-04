import pytest
from fastapi.testclient import TestClient

from monitoramento import api, config


@pytest.fixture()
def cliente_http(monkeypatch, modelo_sintetico):
    monkeypatch.setattr(api.treino, "carregar_modelo", lambda: modelo_sintetico)
    monkeypatch.setattr(
        api.treino,
        "carregar_metadados",
        lambda: {"versao": "teste", "modelo": "HGB", "holdout": {"auc": 0.8}},
    )
    with TestClient(api.app) as c:
        yield c


def _registro(clientes, i=0):
    return {f: float(clientes.iloc[i][f]) for f in config.FEATURES}


def test_health(cliente_http):
    r = cliente_http.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modelo_carregado": True}


def test_predict_retorna_probabilidades(cliente_http, clientes):
    registros = [_registro(clientes, i) for i in range(5)]
    r = cliente_http.post("/predict", json={"registros": registros})
    assert r.status_code == 200
    corpo = r.json()
    assert len(corpo["scores"]) == 5
    assert all(0 <= s <= 1 for s in corpo["scores"])
    assert corpo["versao_modelo"] == "teste"


def test_predict_aceita_nulos(cliente_http, clientes):
    registro = _registro(clientes)
    registro["PAY_0"] = None
    r = cliente_http.post("/predict", json={"registros": [registro]})
    assert r.status_code == 200


def test_payload_invalido_vira_422_e_conta_erro(cliente_http):
    r = cliente_http.post("/predict", json={"registros": [{"LIMIT_BAL": "abc"}]})
    assert r.status_code == 422
    metricas = cliente_http.get("/metrics").text
    assert 'api_erros_total{rota="/predict",status="422"}' in metricas


def test_metrics_formato_prometheus(cliente_http, clientes):
    cliente_http.post("/predict", json={"registros": [_registro(clientes)]})
    r = cliente_http.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    texto = r.text
    assert "api_latencia_segundos_bucket" in texto
    assert "api_score_predicao_count" in texto
    assert 'api_requisicoes_total{rota="/predict",status="200"}' in texto
