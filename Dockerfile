FROM python:3.11-slim

WORKDIR /app

# System deps (build-essential for any wheels) 
# ADDED: libpq-dev for psycopg2/PostgreSQL driver
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]