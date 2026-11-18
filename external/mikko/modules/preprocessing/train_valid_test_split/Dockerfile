FROM continuumio/miniconda3:4.7.12

WORKDIR /home/train_valid_test_split

ADD modules/preprocessing/train_valid_test_split/requirements.txt /home/train_valid_test_split/requirements.txt
RUN conda install --file /home/train_valid_test_split/requirements.txt

COPY modules/preprocessing/train_valid_test_split /home/train_valid_test_split
COPY base_dockex /home/train_valid_test_split/base_dockex

ENV PYTHONPATH /home/train_valid_test_split

ENTRYPOINT ["python", "-u", "/home/train_valid_test_split/train_valid_test_split.py"]