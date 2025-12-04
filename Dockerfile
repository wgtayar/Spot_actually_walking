# Lightweight container for running the Spot examples.
# Uses pip to install Drake and the rest of the Python stack listed in
# underactuated/requirements.txt.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies needed by Drake, matplotlib, and graphviz.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        graphviz \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libglu1-mesa \
        xvfb && \
    rm -rf /var/lib/apt/lists/*

# Some tools expect /usr/bin/python3; base image installs Python in /usr/local.
RUN ln -s /usr/local/bin/python /usr/bin/python3

WORKDIR /workspace/spot

# Install Drake first (use a currently published version), then the rest of the deps.
# The repo's requirements pin an older nightly that is no longer hosted, so we
# filter that line out when installing the rest.
ARG DRAKE_VERSION=1.47.0
COPY underactuated/requirements.txt underactuated/requirements.txt
RUN pip install "drake==${DRAKE_VERSION}" --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly && \
    grep -v '^drake==' underactuated/requirements.txt > /tmp/requirements.nodrake && \
    pip install -r /tmp/requirements.nodrake

# Copy the rest of the workspace.
COPY . .

# Make local packages importable inside the container.
ENV PYTHONPATH="/workspace/spot:/workspace/spot/underactuated"

EXPOSE 7000

CMD ["/bin/bash"]
