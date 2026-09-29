FROM python:3.10-slim

# Install system dependencies required for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
