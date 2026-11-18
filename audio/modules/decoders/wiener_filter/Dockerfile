FROM continuumio/miniconda3:4.7.12

WORKDIR /home/wiener_filter

ADD modules/decoders/wiener_filter/requirements.txt /home/wiener_filter/requirements.txt
RUN conda install --file /home/wiener_filter/requirements.txt

COPY modules/decoders/wiener_filter /home/wiener_filter
COPY base_dockex /home/wiener_filter/base_dockex

ENV PYTHONPATH /home/wiener_filter

ENTRYPOINT ["python", "-u", "/home/wiener_filter/wiener_filter.py"]