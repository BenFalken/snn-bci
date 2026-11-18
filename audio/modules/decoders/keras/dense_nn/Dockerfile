FROM python:3.11-slim

WORKDIR /home/dense_nn

ADD modules/decoders/keras/dense_nn/requirements.txt /home/dense_nn/requirements.txt
RUN pip install -r /home/dense_nn/requirements.txt

COPY modules/decoders/keras/dense_nn /home/dense_nn
COPY base_dockex /home/dense_nn/base_dockex

ENV PYTHONPATH /home/dense_nn

ENTRYPOINT ["python", "-u", "/home/dense_nn/dense_nn.py"]