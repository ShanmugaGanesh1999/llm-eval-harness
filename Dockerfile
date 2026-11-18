FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY evals ./evals
COPY config ./config
COPY data ./data
COPY tests ./tests
COPY pytest.ini .
# Default: EVAL_MODE=offline scores the baseline fixture; EVAL_MODE=live calls P1/P3/P4.
CMD ["python", "-m", "evals"]
