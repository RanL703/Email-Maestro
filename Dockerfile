FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY requirements.app.txt .
RUN pip install --no-cache-dir -r requirements.app.txt

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app.py"]
