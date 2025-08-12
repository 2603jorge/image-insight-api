FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-spa libzbar0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=10000
CMD gunicorn -w 2 -b 0.0.0.0:$PORT app:app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
