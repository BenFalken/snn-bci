FROM continuumio/miniconda3:4.7.12

WORKDIR /home/sklearn_preprocess

ADD modules/preprocessing/sklearn_preprocess/requirements.txt /home/sklearn_preprocess/requirements.txt
RUN conda install --file /home/sklearn_preprocess/requirements.txt

COPY modules/preprocessing/sklearn_preprocess /home/sklearn_preprocess
COPY base_dockex /home/sklearn_preprocess/base_dockex

ENV PYTHONPATH /home/sklearn_preprocess

ENTRYPOINT ["python", "-u", "/home/sklearn_preprocess/sklearn_preprocess.py"]