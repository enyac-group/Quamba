FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

RUN apt-get update && apt-get install -y \
    sudo \
    python3 git-core zip unzip tmux htop \
    curl \
    bash \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
RUN curl -L -s https://repo.anaconda.com/archive/ > anaconda.folder.list.txt
RUN sha256sum Anaconda3-2023.09-0-Linux-x86_64.sh > shasum.txt
RUN bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /usr/local/anaconda3 && \
    rm Anaconda3-2023.09-0-Linux-x86_64.sh && \
    echo ". /usr/local/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc

RUN apt-get update \
    && apt-get -y install build-essential \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.24.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

ENV PATH="/usr/local/anaconda3/bin:${PATH}"
ENV HOME_PATH="/home/user"

RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid user --shell /bin/bash --create-home user --home-dir "${HOME_PATH}" \
  && mkdir -p /etc/sudoers.d \
  && echo 'user ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers.d/user \
  && chmod 440 /etc/sudoers.d/user && \
  echo ". /usr/local/anaconda3/etc/profile.d/conda.sh" >> "${HOME_PATH}/.bashrc"

COPY ./docker-entrypoint.sh /bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

WORKDIR "$HOME_PATH"
CMD ["bash"]

