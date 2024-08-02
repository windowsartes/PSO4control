FROM python:3.10

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk

COPY ./src ./src
COPY ./main.py ./main.py

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./main.py", "./config.json"]