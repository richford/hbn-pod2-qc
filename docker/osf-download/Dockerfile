FROM hbn-pod2/base:conda-tex

LABEL org.opencontainers.image.authors="richiehalford@gmail.com"

RUN pip install osfclient
RUN mkdir -p /home/8cy32/osfstorage
WORKDIR /home/8cy32

COPY base_file_list.txt /base_file_list.txt
COPY nifti_file_list.txt /nifti_file_list.txt
COPY tfrec_file_list.txt /tfrec_file_list.txt
COPY download_files.sh /usr/local/bin/download_files.sh
RUN chmod +x /usr/local/bin/download_files.sh 

ENTRYPOINT ["download_files.sh"]
CMD ["/base_file_list.txt"]