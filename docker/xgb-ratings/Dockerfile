FROM continuumio/miniconda3:4.9.2-alpine

RUN conda config --add channels conda-forge
RUN conda install ipython numpy matplotlib pandas s3fs scikit-learn=0.23.2 scikit-optimize shap tqdm xgboost

COPY xgb-ratings.py /usr/local/bin/xgb-ratings.py
RUN chmod +x /usr/local/bin/xgb-ratings.py 

ENV OUTDATED_IGNORE 1

ENTRYPOINT ["xgb-ratings.py", "/input/expert-ratings/expert_ratings.csv", "/input/fibr-ratings", "/output/xgb-model-dir", "/output", "--random_state", "42"]