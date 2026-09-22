#!/usr/bin/env bash
# Boot sequence for the single-container deployment:
#   1. make sure the bundles are present (download if a source is configured)
#   2. share one API secret between the API and the dashboard
#   3. start uvicorn, streamlit and nginx; exit if any of them dies
set -euo pipefail
# repo root: WebApp/deploy/start.sh -> ../..
cd "$(dirname "$0")/../.."

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/WebApp:$(pwd)/Maternal_Health"

if [ -z "${MCH_API_SECRET:-}" ]; then
  export MCH_API_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "[start] MCH_API_SECRET not set — generated one for this container's lifetime." \
       "Keys will not survive a restart; set MCH_API_SECRET to make them durable."
fi

echo "[start] fetching model bundles…"
python - <<'PY'
from core.model_store import ensure_models
print("[start] bundles ready in", ensure_models(progress=print))
PY

mkdir -p /tmp/nginx/client_body /tmp/nginx/proxy /tmp/nginx/fastcgi /tmp/nginx/uwsgi /tmp/nginx/scgi

echo "[start] starting API on :8000"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 \
  --app-dir WebApp --proxy-headers --log-level info &
API_PID=$!

echo "[start] starting dashboard on :8501"
python -m streamlit run WebApp/streamlit_app.py \
  --server.port 8501 --server.address 127.0.0.1 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  --browser.gatherUsageStats false &
UI_PID=$!

echo "[start] starting nginx on :7860"
nginx -c "$(pwd)/WebApp/deploy/nginx.conf" -g 'daemon off;' &
NGINX_PID=$!

trap 'kill $API_PID $UI_PID $NGINX_PID 2>/dev/null || true' EXIT
wait -n $API_PID $UI_PID $NGINX_PID
echo "[start] a process exited; shutting down"
exit 1
