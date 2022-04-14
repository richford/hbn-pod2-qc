FROM hbn-pod2/base:conda-tex

RUN conda config --add channels conda-forge
COPY package-list.txt /package-list.txt
RUN conda install --file /package-list.txt --yes

COPY xgb-ratings.py /usr/local/bin/xgb-ratings.py
RUN chmod +x /usr/local/bin/xgb-ratings.py 

RUN echo savefig.dpi: 300 >> /tex.mplstyle

ENV OUTDATED_IGNORE 1
LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

ENTRYPOINT ["xgb-ratings.py"]
CMD ["/input/expert-ratings/expert_ratings.csv", \
     "/input/fibr-ratings", \
     "/input/fibr-ratings/saved-xgb-models", \
     "/input/raw-expert-qc", \
     "/output", \
     "/figures", \
     "--random_state", \
     "42"\
    ]