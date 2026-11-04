# Imagem única usada pela API e pelo simulador/monitor.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MONITORAMENTO_RAIZ=/app \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Baixa o dataset público (UCI) e treina o modelo durante o build:
# o container já sobe com modelo, referência e pool de produção prontos.
RUN python -m monitoramento.treino

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000 8001
CMD ["uvicorn", "monitoramento.api:app", "--host", "0.0.0.0", "--port", "8000"]
