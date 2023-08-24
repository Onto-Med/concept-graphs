ARG ROOT_CONTAINER=nvidia/cuda:11.0.3-base-ubuntu20.04
#ARG ROOT_CONTAINER=python:3.8-alpine

FROM $ROOT_CONTAINER

ARG REST_API_WORKDIR=/rest_api
ARG MODEL_DIR="${REST_API_WORKDIR}/models"

WORKDIR $REST_API_WORKDIR

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

ARG PYTHON=python3.10

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa  --yes
    
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    gcc \
    libgcc-9-dev \
    ${PYTHON} \
    ${PYTHON}-dev \
    ${PYTHON}-distutils \
    # - bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    ca-certificates \
    fonts-liberation \
    locales \
    # - pandoc is used to convert notebooks to html files
    #   it's not present in aarch64 ubuntu image, so we install it here
    pandoc \
    # - run-one - a wrapper script that runs no more
    #   than one unique  instance  of  some  command with a unique set of arguments,
    #   we use `run-one-constantly` to support `RESTARTABLE` option
    run-one \
    sudo \
    # - tini is installed as a helpful container entrypoint that reaps zombie
    #   processes and such of the actual executable we want to start, see
    #   https://github.com/krallin/tini#why-tini for details.
    tini \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Install current pip version for python version
curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON}

# Install Git-LFS and download models
RUN apt-get update --yes && \
    apt-get -y install git-lfs && \
    apt-get -y install gcc && \
    apt-get -y install ca-certificates

# Download sources
RUN git clone https://github.com/Onto-Med/concept-graphs


# Install ML, DL & Visualization
RUN ${PYTHON} -m pip install llvmlite --ignore-installed && \
    ${PYTHON} -m pip install numba==0.53.0 yellowbrick umap-learn==0.5.2 --ignore-installed && \
    ${PYTHON} -m pip install tensorflow transformers sentence-transformers datasets && \
    ${PYTHON} -m pip install spacy spacy-lookups-data spacy-transformers && \
    ${PYTHON} -m pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    ${PYTHON} -m pip install altair networkx pyvis umap-learn[plot] && \
    ${PYTHON} -m pip install python-Levenshtein fuzzywuzzy && \
    ${PYTHON} -m pip install bratiaa && \
    ${PYTHON} -m pip install flask && \
    ${PYTHON} -m pip install pyyaml && \
    ${PYTHON} -m pip install scikit-network && \
    ${PYTHON} -m pip install --upgrade requests && \
    ${PYTHON} -m spacy download de_core_news_sm && \
    ${PYTHON} -m spacy download de_dep_news_trf
    
ENTRYPOINT [ "${PYTHON}" ]

WORKDIR "${REST_API_WORKDIR}/concept-graphs"

CMD ["main.py"]
