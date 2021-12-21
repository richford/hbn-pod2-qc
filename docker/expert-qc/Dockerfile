FROM hbn-pod2/base:conda-tex

RUN conda config --add channels conda-forge
COPY package-list.txt /package-list.txt
RUN conda install --file /package-list.txt --yes

COPY expert_qc.py /usr/local/bin/expert_qc.py
RUN chmod +x /usr/local/bin/expert_qc.py 

ENV OUTDATED_IGNORE 1

ENTRYPOINT ["expert_qc.py"]
CMD ["/input", "/derived-data", "/figures"]