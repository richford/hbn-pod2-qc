FROM continuumio/miniconda3:4.9.2-alpine

RUN conda config --add channels conda-forge
RUN conda install dask[dataframe] matplotlib numpy pandas pingouin seaborn scikit-learn

COPY expert_qc.py /usr/local/bin/expert_qc.py
RUN chmod +x /usr/local/bin/expert_qc.py 

ENV OUTDATED_IGNORE 1

ENTRYPOINT ["expert_qc.py"]
CMD ["/input"]