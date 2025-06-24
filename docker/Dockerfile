FROM python:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including redis-server
RUN apt-get update && apt-get install -y \
    postgresql \
    postgresql-contrib \
    redis-server \
    gcc \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install Python dependencies (add redis to requirements.txt)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

USER postgres
# Initialize PostgreSQL, create database and table
RUN service postgresql stop && \
    /etc/init.d/postgresql start && \
    psql --command "CREATE DATABASE restaurant_orders;" && \
    psql -d restaurant_orders --command "CREATE TABLE IF NOT EXISTS orders (id SERIAL PRIMARY KEY, phone_number VARCHAR(15) NOT NULL, order_data JSONB NOT NULL, orderdate TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP, status VARCHAR(20) DEFAULT 'pending');" && \
    psql --command "ALTER USER postgres WITH PASSWORD 'sel33man';" &&\
    /etc/init.d/postgresql stop

USER root

EXPOSE 8000 6379

# Start PostgreSQL, Redis, run menuIndexer.py, then main.py
CMD service postgresql start && \
    service redis-server start && \
    python menuIndexer.py  && \
    python ultravox_main.py

