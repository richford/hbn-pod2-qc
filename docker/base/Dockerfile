FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add -U make
# dependencies for texlive
RUN apk add -U --repository http://dl-3.alpinelinux.org/alpine/edge/main \
    poppler harfbuzz-icu
# zziplib (found in edge/community repository) is a dependency to texlive-luatex
RUN apk add -U --repository http://dl-3.alpinelinux.org/alpine/edge/community \
    zziplib
RUN apk add -U --repository http://dl-3.alpinelinux.org/alpine/edge/testing \
    texlive-full
COPY tex.mplstyle /tex.mplstyle
COPY plot-formatting.py /usr/local/bin/plot_formatting.py