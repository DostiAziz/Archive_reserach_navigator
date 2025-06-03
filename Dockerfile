FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This will now copy data/, logs/, test/, src/, models/, etc.
# But NOT venv/ (excluded by .dockerignore)
COPY . .

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONPATH=/app/src:/app

EXPOSE 8501

CMD ["streamlit", "run", "src/Main.py", "--server.port=8501", "--server.address=0.0.0.0"]