FROM dl.nvidia/cuda:12.8-cudnn9-runtime-torch2.7.1

COPY sdwseg /home/sdwseg/sdwseg
copy util /home/sdwseg/util
COPY whl /home/sdwseg/whl
COPY tool/train.py /home/sdwseg


RUN pip3 install /home/sdwseg/whl/engine_frame-*.whl

WORKDIR /home/sdwseg
ENTRYPOINT ["python3", "train.py"]