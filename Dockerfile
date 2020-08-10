FROM python:3.7

RUN apt-get update

COPY ./requirements.txt /app/requirements.txt

# RUN pip install -r requirements.txt
RUN pip install numpy==1.18.5
RUN pip install opencv-python==4.3.0.36
RUN pip install tensorflow==2.1.0
RUN pip install matplotlib==3.2.2

COPY ./train.py /app/train.py
COPY ./inference.py /app/inference.py
COPY ./haarcascade_frontalface_default.xml /app/haarcascade_frontalface_default.xml
WORKDIR /app

ENTRYPOINT [ "python", "inference.py" ]