FROM continuumio/miniconda3:4.9.2-alpine

RUN conda install pandas tqdm

COPY fb_json_to_csv.py /usr/local/bin/fb_json_to_csv.py
RUN chmod +x /usr/local/bin/fb_json_to_csv.py 

ENTRYPOINT ["fb_json_to_csv.py"]
CMD ["/output"]