FROM continuumio/miniconda3:4.7.12

WORKDIR /home/kalman_filter

ADD modules/decoders/kalman_filter/requirements.txt /home/kalman_filter/requirements.txt
RUN conda install --file /home/kalman_filter/requirements.txt

COPY modules/decoders/kalman_filter /home/kalman_filter
COPY base_dockex /home/kalman_filter/base_dockex

ENV PYTHONPATH /home/kalman_filter

ENTRYPOINT ["python", "-u", "/home/kalman_filter/kalman_filter.py"]