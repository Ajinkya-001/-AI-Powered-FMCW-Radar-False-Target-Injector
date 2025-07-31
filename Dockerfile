FROM python:3.10-slim

WORKDIR /app

# Copy model
COPY ai_module/radar_classifier.h5 /app/radar_classifier.h5

# Copy API main file
COPY api/main.py /app/main.py

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
