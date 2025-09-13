FROM python:3.11-slim-bookworm

RUN pip install poetry==2.1.4

ENV POETRY_NO_INTERACTION=true \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /rest_api

COPY . .

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN poetry install --without test --with rag --no-root && rm -rf $POETRY_CACHE_DIR

ENTRYPOINT [ "waitress-serve", "--port=9007", "main:main_objects.app" ]
