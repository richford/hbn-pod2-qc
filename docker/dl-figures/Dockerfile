FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add --update --no-cache graphviz
RUN conda update -n base -c defaults conda
RUN conda config --add channels conda-forge
RUN conda install dask[dataframe] matplotlib numpy seaborn scikit-learn=0.23.2 tensorflow=2.4.0
RUN conda install pydot

COPY dl-figures.py /usr/local/bin/dl-figures.py
RUN chmod +x /usr/local/bin/dl-figures.py 

ENTRYPOINT ["dl-figures.py"]
CMD ["/input/saved-model", \
     "/input/training-logs", \
     "/input/report-set-predictions", \
     "/figures"\
    ]