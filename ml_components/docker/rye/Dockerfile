FROM pytorch/pytorch:latest as yamamiya-pm-ml-rye

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
ARG UID=${UID} && GID=${GID} && USER_NAME=${USER_NAME} && GROUP_NAME=${GROUP_NAME} && \
    PASSWD=${USER_NAME} && HOME=/home/${USER_NAME}
ENV HOME=/home/${USER_NAME}
RUN groupadd -g 1000 nsakairi && useradd -m -s /bin/bash -u ${UID} -g ${GID} -G sudo ${USER_NAME} && \
    echo ${USER_NAME}:${USER_NAME} | chpasswd && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USER_NAME}

WORKDIR ${HOME}/app
RUN sudo chown -R ${USER_NAME} ${HOME}  && sudo chmod -R 777 ${HOME}

RUN sudo apt update && \
    sudo apt install --no-install-recommends -y python3-pip curl unzip build-essential libgl1-mesa-glx git


### install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install && \ 
    rm ./aws*.* && rm -fR ./aws
ARG MINIO_ENDPOINT_URL=${MINIO_ENDPOINT_URL}
ENV MINIO_ENDPOINT_URL ${MINIO_ENDPOINT_URL}

### Install rye
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"

WORKDIR ${HOME}/workspace
# WORKDIR /workspace

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends build-essential
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

ENV RYE_HOME=${HOME}/rye
ENV PATH=$RYE_HOME/shims:$PATH

RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash
# RUN curl -sSf https://rye.astral.sh/get | bash
# RUN curl -sSfL https://rye-up.com/get | sudo RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash
#RUN curl -sSfL https://rye-up.com/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash

WORKDIR ${HOME}