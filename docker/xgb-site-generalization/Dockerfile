FROM hbn-pod2/base:conda-tex

RUN conda config --add channels conda-forge
COPY package-list.txt /package-list.txt
RUN conda install --file /package-list.txt --yes

COPY xgb-site-generalization.py /usr/local/bin/xgb-site-generalization.py
RUN chmod +x /usr/local/bin/xgb-site-generalization.py 

ENV OUTDATED_IGNORE 1
LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

ENTRYPOINT ["xgb-site-generalization.py"]
CMD ["/input/dl-predictions", \
     "/input/expert-ratings/expert_ratings.csv", \
     "/input/fibr-ratings", \
     "/input/saved-models", \
     "/figures" \
    ]