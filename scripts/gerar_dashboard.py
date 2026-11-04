"""Gera o JSON do dashboard do Grafana (infra/grafana/dashboards/monitoramento.json).

Manter o dashboard como código facilita revisão em PR: rode este script depois de
alterar painéis e versione o JSON gerado.
"""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
DESTINO = Path(__file__).resolve().parents[1] / "infra/grafana/dashboards/monitoramento.json"

_id = 0


def _novo_id() -> int:
    global _id
    _id += 1
    return _id


def alvo(expr: str, legenda: str = "", instantaneo: bool = False, ref: str = "A") -> dict:
    t = {"datasource": DS, "expr": expr, "legendFormat": legenda, "refId": ref}
    if instantaneo:
        t |= {"instant": True, "range": False, "format": "table"}
    return t


def limiares(*passos: tuple[str, float | None]) -> dict:
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in passos]}


def linha(titulo: str, y: int) -> dict:
    return {
        "type": "row",
        "title": titulo,
        "id": _novo_id(),
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "panels": [],
    }


def stat(titulo, expr, x, y, w=4, h=4, unidade="none", decimais=None, th=None, descricao=""):
    return {
        "type": "stat",
        "title": titulo,
        "description": descricao,
        "id": _novo_id(),
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [alvo(expr)],
        "fieldConfig": {
            "defaults": {
                "unit": unidade,
                "decimals": decimais,
                "color": {"mode": "thresholds"},
                "thresholds": th or limiares(("green", None)),
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background",
            "graphMode": "area",
            "textMode": "value",
        },
    }


def serie(
    titulo,
    alvos,
    x,
    y,
    w=12,
    h=8,
    unidade="none",
    th=None,
    estilo_limiar="off",
    descricao="",
    min_=None,
    max_=None,
    legenda_tabela=False,
    max_suave=None,
):
    defaults = {
        "unit": unidade,
        "min": min_,
        "max": max_,
        "color": {"mode": "palette-classic"},
        "custom": {
            "lineWidth": 2,
            "fillOpacity": 5,
            "showPoints": "never",
            "spanNulls": True,
            "thresholdsStyle": {"mode": estilo_limiar},
            "axisSoftMax": max_suave,
        },
        "thresholds": th or limiares(("green", None)),
    }
    return {
        "type": "timeseries",
        "title": titulo,
        "description": descricao,
        "id": _novo_id(),
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": alvos,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {
                "displayMode": "table" if legenda_tabela else "list",
                "placement": "right" if legenda_tabela else "bottom",
                "calcs": ["lastNotNull", "max"] if legenda_tabela else [],
                "sortBy": "Max",
                "sortDesc": True,
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


TH_PSI = limiares(("green", None), ("orange", 0.1), ("red", 0.2))


def construir() -> dict:
    p = []
    y = 0
    p.append(linha("Visão geral", y))
    y += 1
    p += [
        stat(
            "Dia simulado",
            "monitor_dia_simulado",
            0,
            y,
            descricao="Dia da linha do tempo simulada. Drift de covariável a partir do dia 15, "
            "incidente de qualidade nos dias 28-30, drift de conceito a partir do dia 40.",
            th=limiares(("blue", None)),
        ),
        stat(
            "Disponibilidade da API (10 min)",
            'avg_over_time(up{job="api-credito"}[10m])',
            4,
            y,
            unidade="percentunit",
            decimais=2,
            th=limiares(("red", None), ("orange", 0.99), ("green", 0.999)),
        ),
        stat(
            "Latência p95 /predict",
            "histogram_quantile(0.95, sum by (le) "
            '(rate(api_latencia_segundos_bucket{rota="/predict"}[1m])))',
            8,
            y,
            unidade="s",
            th=limiares(("green", None), ("orange", 0.1), ("red", 0.25)),
        ),
        stat(
            "Taxa de erro /predict",
            'sum(rate(api_erros_total{rota="/predict"}[1m])) '
            '/ sum(rate(api_requisicoes_total{rota="/predict"}[1m]))',
            12,
            y,
            unidade="percentunit",
            decimais=1,
            th=limiares(("green", None), ("orange", 0.01), ("red", 0.05)),
        ),
        stat(
            "Maior PSI de feature",
            "max(monitor_psi)",
            16,
            y,
            decimais=3,
            th=TH_PSI,
        ),
        stat(
            "AUC (rótulo atrasado)",
            'monitor_performance{metrica="auc"}',
            20,
            y,
            decimais=3,
            th=limiares(("red", None), ("orange", 0.7), ("green", 0.74)),
        ),
    ]
    y += 4

    p.append(linha("Serviço: latência, volume e erros", y))
    y += 1
    p += [
        serie(
            "Latência de /predict (p50 / p95 / p99)",
            [
                alvo(
                    f"histogram_quantile({q}, sum by (le) "
                    f'(rate(api_latencia_segundos_bucket{{rota="/predict"}}[30s])))',
                    f"p{int(q * 100)}",
                    ref=r,
                )
                for q, r in ((0.5, "A"), (0.95, "B"), (0.99, "C"))
            ],
            0,
            y,
            w=8,
            unidade="s",
        ),
        serie(
            "Requisições por segundo, por status",
            [alvo("sum by (status) (rate(api_requisicoes_total[30s]))", "HTTP {{status}}")],
            8,
            y,
            w=8,
            unidade="reqps",
        ),
        serie(
            "Score servido (quantis da distribuição)",
            [
                alvo(
                    f"histogram_quantile({q}, sum by (le) (rate(api_score_predicao_bucket[30s])))",
                    f"p{int(q * 100)}",
                    ref=r,
                )
                for q, r in ((0.5, "A"), (0.9, "B"))
            ],
            16,
            y,
            w=8,
            unidade="percentunit",
            descricao="Probabilidade de inadimplência prevista. Mudanças aqui afetam "
            "diretamente a taxa de aprovação.",
        ),
    ]
    y += 8

    p.append(linha("Qualidade dos dados de entrada", y))
    y += 1
    p += [
        serie(
            "Taxa de nulos por feature",
            [alvo("monitor_taxa_nulos", "{{feature}}")],
            0,
            y,
            unidade="percentunit",
            th=limiares(("green", None), ("red", 0.05)),
            estilo_limiar="line+area",
            legenda_tabela=True,
            min_=0,
            max_suave=0.1,
        ),
        serie(
            "Taxa de valores fora do intervalo por feature",
            [alvo("monitor_taxa_fora_intervalo", "{{feature}}")],
            12,
            y,
            unidade="percentunit",
            th=limiares(("green", None), ("red", 0.01)),
            estilo_limiar="line+area",
            legenda_tabela=True,
            min_=0,
            max_suave=0.05,
        ),
    ]
    y += 8

    p.append(linha("Drift: PSI e KS contra a referência (holdout do treino)", y))
    y += 1
    p += [
        serie(
            "PSI por feature (0,1 atenção · 0,2 crítico)",
            [alvo("monitor_psi", "{{feature}}")],
            0,
            y,
            w=12,
            h=9,
            th=TH_PSI,
            estilo_limiar="line+area",
            legenda_tabela=True,
            min_=0,
            max_suave=0.3,
            descricao="PSI = Σ (a_i − r_i) · ln(a_i / r_i), com faixas definidas nos decis "
            "da referência.",
        ),
        serie(
            "Estatística D do teste KS por feature",
            [alvo("monitor_ks_estatistica", "{{feature}}")],
            12,
            y,
            w=6,
            h=9,
            min_=0,
        ),
        serie(
            "PSI do score do modelo",
            [alvo("monitor_psi_score", "PSI do score")],
            18,
            y,
            w=6,
            h=9,
            th=TH_PSI,
            estilo_limiar="line+area",
            min_=0,
            descricao="Drift de predição. Note que o drift de conceito (dia 40+) NÃO aparece "
            "aqui: só a performance com rótulo revela.",
        ),
    ]
    y += 9

    p.append(linha("Performance do modelo (rótulos chegam com 5 dias de atraso)", y))
    y += 1
    p += [
        serie(
            "AUC do lote rotulado vs. baseline do holdout",
            [
                alvo('monitor_performance{metrica="auc"}', "AUC do lote", ref="A"),
                alvo("monitor_auc_baseline", "AUC baseline (holdout)", ref="B"),
                alvo("monitor_auc_baseline - 0.05", "limite de alerta (baseline − 0,05)", ref="C"),
            ],
            0,
            y,
            w=10,
            min_=0.5,
            max_=0.9,
        ),
        serie(
            "Gini e KS de separação",
            [
                alvo('monitor_performance{metrica="gini"}', "Gini", ref="A"),
                alvo('monitor_performance{metrica="ks"}', "KS", ref="B"),
            ],
            10,
            y,
            w=7,
            min_=0,
        ),
        serie(
            "Taxa de inadimplência observada",
            [alvo("monitor_taxa_default_observada", "default observado")],
            17,
            y,
            w=7,
            unidade="percentunit",
        ),
    ]
    y += 8

    p.append(linha("Alertas (regras do Prometheus)", y))
    y += 1
    p.append(
        {
            "type": "table",
            "title": "Alertas disparados agora",
            "id": _novo_id(),
            "datasource": DS,
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": y},
            "targets": [alvo('ALERTS{alertstate="firing"}', instantaneo=True)],
            "transformations": [
                {
                    "id": "organize",
                    "options": {
                        "excludeByName": {
                            "Time": True,
                            "Value": True,
                            "__name__": True,
                            "alertstate": True,
                            "instance": True,
                            "job": True,
                        },
                        "renameByName": {
                            "alertname": "Alerta",
                            "severidade": "Severidade",
                            "categoria": "Categoria",
                            "feature": "Feature",
                            "metrica": "Métrica",
                        },
                    },
                }
            ],
            "options": {"showHeader": True},
            "fieldConfig": {"defaults": {}, "overrides": []},
        }
    )

    return {
        "uid": "monitoramento-credito",
        "title": "Monitoramento do modelo de crédito",
        "description": "Latência, disponibilidade, qualidade de dados, drift (PSI/KS) e "
        "performance com rótulo atrasado",
        "tags": ["ml", "drift", "credito"],
        "timezone": "browser",
        "editable": False,
        "graphTooltip": 1,
        "refresh": "5s",
        "schemaVersion": 39,
        "time": {"from": "now-15m", "to": "now"},
        "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m"]},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": p,
        "version": 1,
    }


if __name__ == "__main__":
    DESTINO.write_text(json.dumps(construir(), indent=2, ensure_ascii=False) + "\n")
    print(f"Dashboard gerado em {DESTINO}")
