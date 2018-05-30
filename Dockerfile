FROM nvidia/cuda:9.1-base

WORKDIR /project

RUN apt-get update \
    && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:jonathonf/python-3.6 \
    && apt-get remove -y software-properties-common \
    && apt autoremove -y \
    && apt-get update \
    && apt-get install -y python3.6 \
    && curl -o /tmp/get-pip.py "https://bootstrap.pypa.io/get-pip.py" \
    && python3.6 /tmp/get-pip.py \
    && apt-get remove -y curl \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /project/requirements.txt
RUN pip install --upgrade pip && pip install -r /project/requirements.txt

EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
