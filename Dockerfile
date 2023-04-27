FROM python:3.9

WORKDIR /app

RUN apt update
RUN apt install ffmpeg gcc g++ unzip curl -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install


COPY requirements.txt /app

RUN pip install -r requirements.txt
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY . /app

ENTRYPOINT ["python", "server.py"]