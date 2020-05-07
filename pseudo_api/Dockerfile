FROM continuumio/miniconda3

WORKDIR usr/src/flask_app


ADD environment.yml environment.yml

RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/flair_pseudo_api/bin:$PATH
RUN /bin/bash -c "source activate flair_pseudo_api"
COPY . .
