FROM python:3.12-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
COPY setup.py .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "biosight.app:app", "--host", "0.0.0.0", "--port", "8000"]