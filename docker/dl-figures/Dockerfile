FROM hbn-pod2/base:conda-tex

RUN apk add --update --no-cache graphviz
RUN conda update -n base -c defaults conda
RUN conda config --add channels conda-forge
COPY package-list.txt /package-list.txt
RUN conda install --file /package-list.txt --yes
RUN conda install -c plotly plotly
RUN conda install nilearn

COPY dl-figures.py /usr/local/bin/dl-figures.py
RUN chmod +x /usr/local/bin/dl-figures.py 

LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

ENTRYPOINT ["dl-figures.py"]
CMD ["/input/saved-model", \
     "/input/training-logs", \
     "/input/report-set-predictions", \
     "/input/nifti-dir", \
     "/figures"\
    ]