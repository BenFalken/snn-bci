FROM python:3.11-slim

WORKDIR /home/lsnn

ADD modules/decoders/torch/lsnn/requirements.txt /home/lsnn/requirements.txt
RUN pip install -r /home/lsnn/requirements.txt

COPY modules/decoders/torch/lsnn /home/lsnn
COPY base_dockex /home/lsnn/base_dockex

ENV PYTHONPATH /home/lsnn

ENTRYPOINT ["python", "-u", "/home/lsnn/lsnn.py"]
