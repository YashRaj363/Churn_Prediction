#!/usr/bin/env bash
# Deploy the churn API to Google Cloud Run.
#
# Prereqs: gcloud CLI authenticated (`gcloud auth login`), a GCP project, and
# billing enabled. This script creates the Artifact Registry repo if missing,
# builds the image with Cloud Build, and deploys to Cloud Run.
#
# Usage:
#   PROJECT_ID=my-proj ./deploy/cloudrun.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-churn}"
SERVICE="${SERVICE:-churn-api}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

echo ">> Project: ${PROJECT_ID}  Region: ${REGION}  Service: ${SERVICE}"

gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Create Artifact Registry repo if it doesn't exist.
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  echo ">> Creating Artifact Registry repo ${REPO}"
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker --location="${REGION}"
fi

echo ">> Building image with Cloud Build"
gcloud builds submit --tag "${IMAGE}" .

echo ">> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=4

echo ">> Done. Service URL:"
gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)'
