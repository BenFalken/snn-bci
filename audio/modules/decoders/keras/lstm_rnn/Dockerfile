FROM python:3.11-slim

WORKDIR /home/lstm_rnn

ADD modules/decoders/keras/lstm_rnn/requirements.txt /home/lstm_rnn/requirements.txt
RUN pip install -r /home/lstm_rnn/requirements.txt

COPY modules/decoders/keras/lstm_rnn /home/lstm_rnn
COPY base_dockex /home/lstm_rnn/base_dockex

ENV PYTHONPATH /home/lstm_rnn

ENTRYPOINT ["python", "-u", "/home/lstm_rnn/lstm_rnn.py"]