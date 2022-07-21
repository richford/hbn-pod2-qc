FROM hbn-pod2/base:conda-tex

RUN apk add --update --no-cache git
RUN conda config --add channels conda-forge
RUN conda install pip scikit-learn s3fs
RUN pip install neurocombat-sklearn
RUN pip install afqinsight==0.3.2
RUN pip install dask[dataframe]

COPY bundle-profiles.py /usr/local/bin/bundle-profiles.py
RUN chmod +x /usr/local/bin/bundle-profiles.py 

ENV OUTDATED_IGNORE 1
LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

ENTRYPOINT ["bundle-profiles.py"]
CMD ["/figures"]
CMD ["/figures", \
     "/input/report-set-predictions"\
    ]