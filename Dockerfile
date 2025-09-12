FROM python:3.11-buster as builder

RUN pip install poetry==2.1.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /rest_api

COPY pyproject.toml poetry.lock ./

RUN poetry install --without test --with rag --no-root && rm -rf $POETRY_CACHE_DIR

FROM nvidia/cuda:11.0.3-base-ubuntu20.04 as runtime

ENV VIRTUAL_ENV=/rest_api/.venv \
    PATH="/rest_api/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . .

ENTRYPOINT [ "waitress-serve", "--port=9007", "main:main_objects.app" ]
