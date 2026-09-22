# Maternal & Child Health Risk — dashboard + REST API in one container.
#
#   docker build -t mch .
#   docker run -p 7860:7860 -e MCH_MODEL_REPO=<hf-user>/<repo> -e MCH_API_SECRET=<secret> mch
#
# Everything it serves lives in WebApp/; Maternal_Health/ supplies src/,
# results/ and images/. nginx on :7860 routes /api/* to FastAPI (uvicorn :8000) and everything
# else to Streamlit (:8501). This is the layout Hugging Face Spaces expects
# (single exposed port, non-root user 1000), and it works unchanged on any
# Docker host.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user \
 && mkdir -p /var/lib/nginx /var/log/nginx /tmp/nginx \
 && chown -R user:user /var/lib/nginx /var/log/nginx /tmp/nginx

WORKDIR /app
COPY --chown=user WebApp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN chmod +x WebApp/deploy/start.sh && mkdir -p Maternal_Health/models && chown -R user:user /app

USER user
ENV HOME=/home/user \
    MCH_API_PROXIED=1 \
    MCH_API_ROOT_PATH=/api \
    MCH_API_INTERNAL_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -sf http://127.0.0.1:7860/api/health || exit 1

CMD ["WebApp/deploy/start.sh"]
