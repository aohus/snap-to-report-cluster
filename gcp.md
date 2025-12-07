# 1. GCP 로그인

# gcloud auth login

# # 2. 프로젝트 설정 (PROJECT_ID를 본인의 프로젝트 ID로 변경)

# export PROJECT_ID="your-gcp-project-id"

# gcloud config set project $PROJECT_ID

# # 3. 필요한 API 활성화

# gcloud services enable run.googleapis.com \

# artifactregistry.googleapis.com \

# cloudbuild.googleapis.com

# # 4. Artifact Registry 저장소 생성 (Docker 이미지 저장소)

# # 리전은 한국(asia-northeast3) 추천

# gcloud artifacts repositories create cluster-repo \

# --repository-format=docker \

# --location=asia-northeast3 \

# --description="Cluster Backend Repository"
