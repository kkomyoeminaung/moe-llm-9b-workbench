# Build stage for frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Final stage
FROM python:3.10-slim
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend
COPY --from=frontend-builder /app/dist /app/frontend/dist

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Run the app
CMD ["uvicorn", "backend.app_complete:app", "--host", "0.0.0.0", "--port", "8080"]
