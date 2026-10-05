FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.14.1 \
    "sqlalchemy>=1.4.36,<2.0" \
    boto3==1.34.144 \
    psycopg2-binary==2.9.9

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=10s --retries=10 \
    CMD curl -f http://localhost:5000/health || exit 1
