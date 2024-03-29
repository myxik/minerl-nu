FROM myxik/myxik_container:latest
RUN apt update && apt install -y curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \
    xvfb \
    python-opengl \
    ffmpeg \
    x11-xserver-utils tightvncserver

RUN apt install -y openjdk-8-jdk

RUN pip install gym
RUN pip install minerl

RUN apt update && apt install -y cmake libopenmpi-dev python3-dev zlib1g-dev

RUN pip install stable-baselines3[extra]

ENV PYTHONPATH "$(PYTHONPATH):/workspace"