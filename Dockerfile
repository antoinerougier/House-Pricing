FROM python:3.12-slim

WORKDIR /app

# Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste de l'application
COPY src/ ./src/
COPY app.py .
COPY model.joblib .
COPY scaler.joblib .
COPY static/ ./static/

EXPOSE 8000

# Utilise Uvicorn pour lancer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]