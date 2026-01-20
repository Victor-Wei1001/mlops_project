FROM python:3.11-slim

WORKDIR /app

# 1. 安装 Git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 复制项目 (增加对隐藏文件夹的显式复制)
COPY . .
# 这一行是“双保险”，强行确保隐藏文件夹存在
COPY .dvc /app/.dvc 

# 4. 设置凭证
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/dtu-mlops-project-484513-db6b7e34022c.json

CMD ["python", "src/models/train_model.py"]