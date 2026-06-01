FROM python:3.11-slim-bookworm

RUN pip install uv

WORKDIR /rest_api

# Tests and Jupyter notebooks are excluded from the build context via .dockerignore.
COPY . .

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN uv sync --frozen --no-group test && uv cache clean

# Dependencies are installed at build time. --no-sync prevents uv from
# resolving/downloading packages again when the container starts.
ENTRYPOINT [ "uv", "run", "--no-sync", "waitress-serve", "--call", "--port=9007", "main:create_app" ]
