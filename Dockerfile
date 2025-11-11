
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ports: HF Spaces uses $PORT, other platforms can map to 8050 by default
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
