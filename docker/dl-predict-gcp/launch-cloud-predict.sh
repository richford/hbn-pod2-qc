#!/bin/sh
JOB_NAME="${1:-$(echo $RANDOM | md5 | head -c20; echo;)}"

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

MIN_SEED=0
MAX_SEED=9
for seed in $(seq ${MIN_SEED} ${MAX_SEED}); do
    gcloud ai-platform jobs submit training ${JOB_NAME}_predict7_seed${seed} \
        --staging-bucket=gs://${BUCKET_NAME} \
        --package-path=predictmodel \
        --module-name=predictmodel.predict_tensorfa \
        --runtime-version=2.4 \
        --python-version=3.7 \
        --config=../config.yaml \
        --region=us-central1 \
        --project ${PROJECT_ID} \
        -- \
        --gcs_bucket=${BUCKET_NAME} \
        --job_name=${JOB_NAME} \
        --n_channels=5 \
        --dataset_name=b0-tensorfa-dwiqc \
        --dataset_seed=${seed} \
        --model_loss=binary_crossentropy \
        --compute-volume-numbers
done

for seed in $(seq ${MIN_SEED} ${MAX_SEED}); do
    gcloud ai-platform jobs describe ${JOB_NAME}_predict7_seed${seed}
done