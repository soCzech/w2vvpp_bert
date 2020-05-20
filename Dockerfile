FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install regex Cython
RUN pip install fairseq
