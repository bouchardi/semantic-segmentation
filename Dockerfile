FROM python:3.6.5-stretch

WORKDIR /project

COPY requirements.txt /project/requirements.txt
RUN pip install --upgrade pip && pip install -r /project/requirements.txt

EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
