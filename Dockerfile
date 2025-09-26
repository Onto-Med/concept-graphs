FROM python:3.11-slim-bookworm

RUN pip install uv

WORKDIR /rest_api

COPY . .

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN uv sync --no-group test && uv cache clean

ENTRYPOINT [ "uv", "run", "waitress-serve", "--port=9007", "main:main_objects.app" ]
