# Base image comes with PyTorch, numpy, flash-attn
ARG BASE
FROM ${BASE} as stable

# Install other dependencies, but not the source code.
COPY pyproject.toml .
COPY src/flexolmo/__init__.py src/flexolmo/__init__.py
COPY src/flexolmo/version.py src/flexolmo/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y flexolmo && \
    rm -rf *

WORKDIR /app/flexolmo

FROM stable as nightly

ARG NIGHTLY_VERSION
RUN pip install --no-cache-dir --pre torch==${NIGHTLY_VERSION}
