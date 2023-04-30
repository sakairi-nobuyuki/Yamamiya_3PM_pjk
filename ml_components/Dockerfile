FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04 as cuda-cudnn-base

# setting Timezone, Launguage

ENV LANG en_US.UTF-8
ENV TZ Asia/Tokyo
ENV DEBIAN_FRONTEND noninteractive

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
RUN apt update && \
  apt install -y --no-install-recommends locales sudo software-properties-common tzdata && \
  locale-gen en_US en_US.UTF-8 && \
  update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
  add-apt-repository universe

# Add user and group
ARG UID=${UID} && GID=${GID} && USER_NAME=${USER_NAME} && GROUP_NAME=${GROUP_NAME}
ARG PASSWD=${USER_NAME} && HOME=/home/${USER_NAME}
RUN groupadd -g 1000 nsakairi && useradd -m -s /bin/bash -u ${UID} -g ${GID} -G sudo ${USER_NAME} && \
    echo ${USER_NAME}:${PASSWD} | chpasswd && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

FROM cuda-cudnn-base AS cuda-cudnn-python-base


# Installing python
ARG USER_NAME=${USER_NAME}
ENV HOME /home/${USER_NAME}
ENV POETRY_VERSION 1.3.1
ENV POETRY_PATH ${HOME}
ENV PATH $PATH:$HOME/.poetry/bin:$HOME/.local/bin:$HOME/bin:$PATH
USER ${USER_NAME}
WORKDIR ${HOME}/app
RUN sudo chown -R ${USER_NAME} ${HOME}  && sudo chmod -R 777 ${HOME}

RUN sudo apt update && \
    sudo apt install --no-install-recommends -y python3.10 python3-pip python3.10-dev unzip \
    python3-setuptools python3-distutils curl &&\
    sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.10 1 && \
    sudo pip install --upgrade pip

### Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python -
RUN echo ${USER_NAME} && echo ${HOME} && ls -la && pwd

### install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install
ARG MINIO_ENDPOINT_URL=${MINIO_ENDPOINT_URL}
ENV MINIO_ENDPOINT_URL ${MINIO_ENDPOINT_URL}

FROM cuda-cudnn-python-base AS yamamiya-pm-ml

ARG USER_NAME=${USER_NAME}
ENV HOME /home/${USER_NAME}

COPY ./pyproject.toml ${HOME}/app 
COPY ./poetry.lock ${HOME}/app

### install python dependencies
RUN poetry config virtualenvs.create false && \
    poetry config installer.parallel false && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip3 install -r requirements.txt --user --no-deps

CMD ["/bin/bash"]