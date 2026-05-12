FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx

COPY main.py .

EXPOSE 8181

CMD ["python", "main.py"]
