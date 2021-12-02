FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add --update --no-cache graphviz
RUN conda update -n base -c defaults conda
RUN conda config --add channels conda-forge
COPY package-list.txt /package-list.txt
RUN conda install --file /package-list.txt --yes

COPY dl-figures.py /usr/local/bin/dl-figures.py
RUN chmod +x /usr/local/bin/dl-figures.py 

ENTRYPOINT ["dl-figures.py"]
CMD ["/input/saved-model", \
     "/input/training-logs", \
     "/input/report-set-predictions", \
     "/figures"\
    ]