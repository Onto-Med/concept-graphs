ARG ROOT_CONTAINER=nvidia/cuda:11.0.3-base-ubuntu20.04
#ARG ROOT_CONTAINER=python:3.10-alpine

FROM $ROOT_CONTAINER

ARG REST_API_WORKDIR=/rest_api

WORKDIR $REST_API_WORKDIR

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON=python3.11
ARG POETRY=/root/.local/bin/poetry
ENV PYTHON=${PYTHON}

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa  --yes

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    gcc \
    libgcc-9-dev \
    g++ \
    ${PYTHON} \
    ${PYTHON}-dev \
    ${PYTHON}-distutils \
    fonts-liberation \
    locales \
    # - tini is installed as a helpful container entrypoint that reaps zombie
    #   processes and such of the actual executable we want to start, see
    #   https://github.com/krallin/tini#why-tini for details.
    tini \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

COPY pyproject.toml ${REST_API_WORKDIR}
RUN curl -sSL https://install.python-poetry.org | ${PYTHON} - && \
    ${POETRY} env use ${PYTHON} && \
    ${POETRY} lock && \
    ${POETRY} install

COPY . .

RUN ${POETRY} run python -m download_models
#    ${PYTHON} -m spacy download de_core_news_sm && \
#    ${PYTHON} -m spacy download de_dep_news_trf && \

CMD [ "waitress-serve", "--port=9007", "main:main_objects.app" ]
