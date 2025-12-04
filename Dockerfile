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

# IMPORTANT: build context is the basil root (see docker-compose.yml)
WORKDIR /workspace/basil

# Install Drake first, then the rest of the deps from the *canonical* underactuated repo.
ARG DRAKE_VERSION=1.47.0

# Copy the requirements from external/underactuated
COPY external/underactuated/requirements.txt /tmp/underactuated_requirements.txt

RUN pip install "drake==${DRAKE_VERSION}" --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly && \
    grep -v '^drake==' /tmp/underactuated_requirements.txt > /tmp/requirements.nodrake && \
    pip install -r /tmp/requirements.nodrake

# Copy the rest of the basil workspace into the image
COPY . /workspace/basil

# Work from the Spot repo by default
WORKDIR /workspace/basil/external/spot

# Make both Spot and Underactuated importable
ENV PYTHONPATH="/workspace/basil/external/spot:/workspace/basil/external/underactuated"

EXPOSE 7000

CMD ["/bin/bash"]
