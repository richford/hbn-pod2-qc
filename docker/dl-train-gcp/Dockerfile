FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add --update --no-cache curl openssh git
WORKDIR /usr/local/bin

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-350.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-sdk-350.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh --quiet

ENV PATH="/usr/local/bin/google-cloud-sdk/bin:$PATH"
RUN gcloud components install beta --quiet

COPY trainmodel /home/trainmodel
COPY config.yaml /home/config.yaml

WORKDIR /home/trainmodel

EXPOSE 9005

COPY launch-cloud-train.sh /usr/local/bin/launch-cloud-train.sh
RUN chmod +x /usr/local/bin/launch-cloud-train.sh

ENTRYPOINT ["launch-cloud-train.sh"]