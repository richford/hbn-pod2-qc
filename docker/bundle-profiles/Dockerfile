FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add --update --no-cache git
RUN conda config --add channels conda-forge
# RUN conda install pip numpy matplotlib seaborn pandas tqdm scikit-learn
RUN conda install pip scikit-learn s3fs
RUN pip install neurocombat-sklearn
RUN pip install git+git://github.com/richford/AFQ-Insight.git@a696621efbe1825ee55779e0ac686e111f09421b
# COPY package-list.txt /package-list.txt
# RUN conda install --file /package-list.txt --yes

COPY bundle-profiles.py /usr/local/bin/bundle-profiles.py
RUN chmod +x /usr/local/bin/bundle-profiles.py 

ENV OUTDATED_IGNORE 1

ENTRYPOINT ["bundle-profiles.py"]
CMD ["/figures"]