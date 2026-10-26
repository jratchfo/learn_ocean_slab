FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY config /app/config

RUN pip install --no-cache-dir .

CMD ["python", "-c", "from learn_ocean_slab import load_config; print(load_config('config/settings.yaml'))"]
