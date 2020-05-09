FROM continuumio/miniconda3

RUN apt-get update
RUN mkdir /app
COPY . /app
WORKDIR /app

RUN conda env create -f environment.yml
EXPOSE 5000
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "app.py"]