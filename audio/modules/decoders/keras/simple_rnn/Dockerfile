FROM python:3.11-slim

WORKDIR /home/simple_rnn

ADD modules/decoders/keras/simple_rnn/requirements.txt /home/simple_rnn/requirements.txt
RUN pip install -r /home/simple_rnn/requirements.txt

COPY modules/decoders/keras/simple_rnn /home/simple_rnn
COPY base_dockex /home/simple_rnn/base_dockex

ENV PYTHONPATH /home/simple_rnn

ENTRYPOINT ["python", "-u", "/home/simple_rnn/simple_rnn.py"]