#!/usr/bin/env bash
# Captura um screenshot do dashboard do Grafana com Chromium headless.
# Uso: CHROME=/caminho/do/chrome scripts/capturar_dashboard.sh [saida.png]
# Requer a stack no ar (docker compose up -d) e o acesso anônimo de leitura do Grafana.
set -euo pipefail

CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || true)}"
SAIDA="${1:-docs/dashboard.png}"
URL="${GRAFANA_URL:-http://localhost:3000}/d/monitoramento-credito/monitoramento-do-modelo-de-credito?orgId=1&kiosk&from=now-6m&to=now"

if [[ -z "$CHROME" ]]; then
  echo "Chromium não encontrado. Defina CHROME=/caminho/do/chrome." >&2
  exit 1
fi

"$CHROME" --headless=new --no-sandbox --disable-gpu --hide-scrollbars \
  --window-size=1800,2000 --virtual-time-budget=20000 \
  --screenshot="$SAIDA" "$URL"
echo "Screenshot salvo em $SAIDA"
