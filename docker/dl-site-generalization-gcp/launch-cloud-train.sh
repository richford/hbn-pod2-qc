#!/bin/sh
JOB_NAME=$1

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
MAX_SEED=0
for seed in $(seq ${MIN_SEED} ${MAX_SEED}); do
    gcloud ai-platform jobs submit training ${JOB_NAME}_train_RUCUNY_test_CBIC_${seed} \
        --staging-bucket=gs://${BUCKET_NAME} \
        --package-path=trainsitemodel \
        --module-name=trainsitemodel.train_tensorfa \
        --runtime-version=2.5 \
        --python-version=3.7 \
        --config=../config.yaml \
        --region=us-central1 \
        --project ${PROJECT_ID} \
        -- \
        --gcs_bucket=${BUCKET_NAME} \
        --job_name=${JOB_NAME} \
        --train_site RU \
        --train_site CUNY \
        --test_site CBIC \
        --n_epochs=100 \
        --n_channels=5 \
        --dataset_name=b0-tensorfa-dwiqc \
        --dataset_seed=${seed} \
        --model_loss=binary_crossentropy \
        --compute-volume-numbers

    # gcloud ai-platform jobs submit training ${JOB_NAME}_train_RU_test_CUNYCBIC_${seed} \
    #     --staging-bucket=gs://${BUCKET_NAME} \
    #     --package-path=trainsitemodel \
    #     --module-name=trainsitemodel.train_tensorfa \
    #     --runtime-version=2.4 \
    #     --python-version=3.7 \
    #     --config=../config.yaml \
    #     --region=us-central1 \
    #     --project ${PROJECT_ID} \
    #     -- \
    #     --gcs_bucket=${BUCKET_NAME} \
    #     --job_name=${JOB_NAME} \
    #     --train_site RU \
    #     --test_site CUNY \
    #     --test_site CBIC \
    #     --n_epochs=100 \
    #     --n_channels=5 \
    #     --dataset_name=b0-tensorfa-dwiqc \
    #     --dataset_seed=${seed} \
    #     --model_loss=binary_crossentropy \
    #     --no-compute-volume-numbers

    # gcloud ai-platform jobs submit training ${JOB_NAME}_train_CBIC_test_RUCUNY_${seed} \
    #     --staging-bucket=gs://${BUCKET_NAME} \
    #     --package-path=trainsitemodel \
    #     --module-name=trainsitemodel.train_tensorfa \
    #     --runtime-version=2.4 \
    #     --python-version=3.7 \
    #     --config=../config.yaml \
    #     --region=us-central1 \
    #     --project ${PROJECT_ID} \
    #     -- \
    #     --gcs_bucket=${BUCKET_NAME} \
    #     --job_name=${JOB_NAME} \
    #     --train_site CBIC \
    #     --test_site RU \
    #     --test_site CUNY \
    #     --n_epochs=100 \
    #     --n_channels=5 \
    #     --dataset_name=b0-tensorfa-dwiqc \
    #     --dataset_seed=${seed} \
    #     --model_loss=binary_crossentropy \
    #     --no-compute-volume-numbers

    # gcloud ai-platform jobs submit training ${JOB_NAME}_train_CBICCUNY_test_RU_${seed} \
    #     --staging-bucket=gs://${BUCKET_NAME} \
    #     --package-path=trainsitemodel \
    #     --module-name=trainsitemodel.train_tensorfa \
    #     --runtime-version=2.4 \
    #     --python-version=3.7 \
    #     --config=../config.yaml \
    #     --region=us-central1 \
    #     --project ${PROJECT_ID} \
    #     -- \
    #     --gcs_bucket=${BUCKET_NAME} \
    #     --job_name=${JOB_NAME} \
    #     --train_site CBIC \
    #     --train_site CUNY \
    #     --test_site RU \
    #     --n_epochs=100 \
    #     --n_channels=5 \
    #     --dataset_name=b0-tensorfa-dwiqc \
    #     --dataset_seed=${seed} \
    #     --model_loss=binary_crossentropy \
    #     --no-compute-volume-numbers
done
