FROM hbn-pod2/base:conda-tex

LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

RUN pip install osfclient
WORKDIR /home
RUN mkdir -p /home/8cy32/osfstorage

ENTRYPOINT ["osf", "-p", "8cy32", "clone", "-U"]
