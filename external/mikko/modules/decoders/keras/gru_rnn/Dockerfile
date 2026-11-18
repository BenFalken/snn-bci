FROM python:3.11-slim

WORKDIR /home/gru_rnn

ADD modules/decoders/keras/gru_rnn/requirements.txt /home/gru_rnn/requirements.txt
RUN pip install -r /home/gru_rnn/requirements.txt

COPY modules/decoders/keras/gru_rnn /home/gru_rnn
COPY base_dockex /home/gru_rnn/base_dockex

ENV PYTHONPATH /home/gru_rnn

ENTRYPOINT ["python", "-u", "/home/gru_rnn/gru_rnn.py"]