FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add --update --no-cache git
RUN conda config --add channels conda-forge
RUN conda install pip scikit-learn s3fs xgboost
RUN pip install neurocombat-sklearn afqinsight

COPY inference.py /usr/local/bin/inference.py
RUN chmod +x /usr/local/bin/inference.py 

ENV OUTDATED_IGNORE 1

ENTRYPOINT ["inference.py"]
CMD ["/inference", "/figures"]