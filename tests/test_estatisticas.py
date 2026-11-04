import numpy as np
import pytest

from monitoramento import estatisticas as est


def test_psi_de_proporcoes_calculado_a_mao():
    ref = np.array([0.5, 0.5])
    atual = np.array([0.6, 0.4])
    esperado = (0.6 - 0.5) * np.log(0.6 / 0.5) + (0.4 - 0.5) * np.log(0.4 / 0.5)
    assert est.psi_de_proporcoes(ref, atual) == pytest.approx(esperado)
    assert esperado == pytest.approx(0.0405, abs=1e-4)


def test_psi_e_simetrico_e_nao_negativo():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    atual = np.array([0.25, 0.25, 0.25, 0.25])
    assert est.psi_de_proporcoes(ref, atual) == pytest.approx(est.psi_de_proporcoes(atual, ref))
    assert est.psi_de_proporcoes(ref, atual) > 0


def test_psi_mesma_distribuicao_e_proximo_de_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 50_000)
    atual = rng.normal(0, 1, 50_000)
    assert est.psi(ref, atual) < 0.01


def test_psi_identico_e_zero():
    x = np.random.default_rng(1).normal(size=1000)
    assert est.psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_normal_deslocada_aproxima_divergencia_de_jeffreys():
    # Para N(0,1) vs N(mu,1), KL(p||q) + KL(q||p) = mu^2. Com 10 faixas de quantis
    # o PSI fica um pouco abaixo do valor contínuo (a discretização perde informação).
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 200_000)
    atual = rng.normal(0.5, 1, 200_000)
    valor = est.psi(ref, atual)
    assert 0.20 < valor < 0.25


def test_psi_cresce_com_o_deslocamento():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 20_000)
    valores = [est.psi(ref, rng.normal(mu, 1, 20_000)) for mu in (0.1, 0.3, 0.6, 1.0)]
    assert valores == sorted(valores)
    assert valores[0] < 0.1 < valores[2]


def test_psi_ignora_nulos():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=10_000)
    atual = rng.normal(size=10_000)
    atual[:3000] = np.nan
    assert est.psi(ref, atual) < 0.02


def test_psi_lote_todo_nulo_e_nan():
    assert np.isnan(est.psi(np.arange(100.0), np.full(10, np.nan)))


def test_faixas_discretas_uma_por_valor():
    faixas = est.calcular_faixas(np.array([-2, -1, 0, 0, 1, 2, 2, 8]))
    # 6 valores distintos -> 6 faixas -> 7 bordas
    assert len(faixas) == 7
    assert faixas[0] == -np.inf and faixas[-1] == np.inf


def test_psi_detecta_categoria_nova():
    ref = np.array([1, 2, 3] * 1000, dtype=float)
    atual = np.array([1, 2, 3] * 300 + [3] * 600, dtype=float)
    assert est.psi(ref, atual) > 0.2


def test_ks_detecta_deslocamento():
    rng = np.random.default_rng(5)
    d_igual, p_igual = est.teste_ks(rng.normal(size=2000), rng.normal(size=2000))
    d_diff, p_diff = est.teste_ks(rng.normal(size=2000), rng.normal(0.5, 1, 2000))
    assert d_diff > d_igual
    assert p_diff < 0.001 < p_igual


def test_metricas_performance_modelo_perfeito_e_aleatorio():
    y = np.array([0, 0, 1, 1])
    perfeito = est.metricas_performance(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfeito == {"auc": 1.0, "gini": 1.0, "ks": 1.0}
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 5000)
    aleatorio = est.metricas_performance(y, rng.random(5000))
    assert abs(aleatorio["auc"] - 0.5) < 0.03
    assert abs(aleatorio["gini"]) < 0.06


def test_metricas_performance_uma_classe_so():
    r = est.metricas_performance(np.zeros(10), np.random.default_rng(0).random(10))
    assert all(np.isnan(v) for v in r.values())
