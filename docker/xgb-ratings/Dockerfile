FROM continuumio/miniconda3:4.9.2-alpine

RUN conda config --add channels conda-forge
RUN conda install dask[dataframe] ipython numpy matplotlib pandas pingouin s3fs scikit-learn=0.23.2 scikit-optimize shap tqdm xgboost

COPY xgb-ratings.py /usr/local/bin/xgb-ratings.py
RUN chmod +x /usr/local/bin/xgb-ratings.py 

ENV OUTDATED_IGNORE 1

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