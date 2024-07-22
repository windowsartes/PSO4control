FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

CMD ["python", "./main.py", "./config.json"]