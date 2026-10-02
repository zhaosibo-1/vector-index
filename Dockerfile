FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY web/ ./web/

# .env 不进镜像：配置只应从环境变量或挂载进来
RUN [ ! -f .env ] && echo "no .env baked into image" || false

EXPOSE 8131
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8131"]
