FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY streamlit_app.py .

ENV PORT=8000
EXPOSE 8000

# ANTHROPIC_API_KEY is optional at runtime -- pass with `-e ANTHROPIC_API_KEY=...`
# to enable real Claude reasoning; otherwise the offline heuristic is used.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
