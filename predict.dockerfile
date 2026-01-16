FROM python:3.11-slim

WORKDIR /app

# 1. 安装 Git (DVC 必须)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. 安装依赖 (此时会安装我们刚加进去的 pathspec 和 dvc)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 复制项目
COPY . .

# 4. 设置凭证
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/dtu-mlops-project-484513-db6b7e34022c.json

 
CMD ["python", "src/models/train_model.py"]