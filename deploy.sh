
#!/bin/bash
PROJECT_ID=$(gcloud config get-value project)
REGION="asia-northeast3"
REPO_NAME="cluster-repo"
IMAGE_NAME="cluster-backend-v2"
SERVICE_NAME="cluster-backend"
# 1. Cloud Build를 사용하여 이미지 빌드 및 푸시
# (로컬에서 빌드하지 않고 구글 클라우드 서버에서 빌드하여 로컬 자원 절약)
echo "🚀 Building container image..."
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest" .

# 2. Cloud Run에 배포
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest" \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 1

# 옵션 설명:
# --memory 4Gi: AI 모델 로딩을 위해 최소 4GB 권장 (무료 등급 내)
# --cpu 2: 처리 속도 향상
# --timeout 300: 클러스터링 작업이 길어질 수 있으므로 타임아웃 5분(300초) 설정
# --concurrency 1: 딥러닝 모델은 보통 쓰레드 안전하지 않거나 CPU를 독점하므로 1로 설정하여 요청 격리