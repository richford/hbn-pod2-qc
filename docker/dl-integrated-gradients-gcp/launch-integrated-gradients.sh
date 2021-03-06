#!/bin/sh
JOB_NAME="${1:-$(echo $RANDOM | md5 | head -c20; echo;)}"
TARGET_CLASS="${2:-1}"

export GOOGLE_APPLICATION_CREDENTIALS=/home/google_application_credentials/${GOOGLE_APPLICATION_CREDENTIALS_FILE}

gcloud init --project ${PROJECT_ID}
gcloud config set project ${PROJECT_ID}

gcloud beta services identity create --service tpu.googleapis.com --project ${PROJECT_ID} 2> svc_account.txt
export SVC_ACCOUNT=$(cut -d ":" -f 2 < svc_account.txt | tr -d " ")
rm svc_account.txt
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member serviceAccount:${SVC_ACCOUNT} --role roles/ml.serviceAgent > /dev/null 2>&1
gsutil acl ch -u ${SVC_ACCOUNT}:OWNER gs://${BUCKET_NAME}
gsutil acl ch -u ${TPU_SERVICE_ACCOUNT}:OWNER gs://${BUCKET_NAME}

echo Using google account $(gcloud config get-value account)
echo Using project $(gcloud config get-value project)
echo Using TPU service accounts ${SVC_ACCOUNT} and ${TPU_SERVICE_ACCOUNT}

for cc in true_pos true_neg false_pos false_neg; do
    gcloud ai-platform jobs submit training ${JOB_NAME}_${cc} \
        --staging-bucket=gs://${BUCKET_NAME} \
        --package-path=ig \
        --module-name=ig.integrated_gradients \
        --runtime-version=2.4 \
        --python-version=3.7 \
        --config=../config.yaml \
        --region=us-central1 \
        --project ${PROJECT_ID} \
        -- \
        --gcs_bucket=${BUCKET_NAME} \
        --n_channels=5 \
        --dataset_name=b0-tensorfa-dwiqc \
        --model_dir=b0_tensorfa_dwiqc \
        --dataset_seed=8 \
        --confusion_class=$cc \
        --target_class=${TARGET_CLASS}
done

for cc in true_pos true_neg false_pos false_neg; do
    gcloud ai-platform jobs describe ${JOB_NAME}_${cc}
done