FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 버퍼링 비활성화 (로그 즉시 출력), 포트 설정
ENV PYTHONUNBUFFERED=1 \
    PORT=8001
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
COPY app_v2 ./app_v2

# 실행 권한 부여 (선택 사항) 및 사용자 변경 (보안)
# Cloud Run은 기본적으로 루트로 실행되지만, 보안을 위해 사용자 변경 권장
# RUN useradd -m myuser
# USER myuser

# 8. 서버 실행 커맨드
# Cloud Run은 $PORT 환경변수(기본 8080)로 들어오는 요청을 받아야 함
CMD ["sh", "-c", "uvicorn app_v2.main:app --host 0.0.0.0 --port ${PORT}"]