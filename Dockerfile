FROM python:3.10

WORKDIR /pso_workdir

COPY . .

RUN pip3 install .

CMD ["/bin/bash"]