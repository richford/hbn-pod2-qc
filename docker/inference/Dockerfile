FROM hbn-pod2/base:conda-tex

RUN apk add --update --no-cache git
RUN conda config --add channels conda-forge
RUN conda install pip scikit-learn s3fs xgboost
RUN pip install neurocombat-sklearn afqinsight

COPY inference.py /usr/local/bin/inference.py
RUN chmod +x /usr/local/bin/inference.py 

ENV OUTDATED_IGNORE 1
LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

ENTRYPOINT ["inference.py"]
CMD ["/inference", "/figures"]