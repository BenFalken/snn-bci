FROM python:3.9-slim

WORKDIR /home/librosa_mel_transformer

COPY modules/tasks/auditory/librosa_mel_transformer/requirements.txt /home/librosa_mel_transformer/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libhdf5-dev && \
    pip install --no-cache-dir -r /home/librosa_mel_transformer/requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY modules/tasks/auditory/librosa_mel_transformer /home/librosa_mel_transformer
COPY base_dockex /home/librosa_mel_transformer/base_dockex

ENV PYTHONPATH=/home/librosa_mel_transformer

ENTRYPOINT ["python", "-u", "/home/librosa_mel_transformer/librosa_mel_transformer.py"]