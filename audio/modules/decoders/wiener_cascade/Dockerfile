FROM continuumio/miniconda3:4.7.12

WORKDIR /home/wiener_cascade

ADD modules/decoders/wiener_cascade/requirements.txt /home/wiener_cascade/requirements.txt
RUN conda install --file /home/wiener_cascade/requirements.txt

COPY modules/decoders/wiener_cascade /home/wiener_cascade
COPY base_dockex /home/wiener_cascade/base_dockex

ENV PYTHONPATH /home/wiener_cascade

ENTRYPOINT ["python", "-u", "/home/wiener_cascade/wiener_cascade.py"]