# FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04 as cuda-cudnn-base
FROM ubuntu:22.04 as ubuntu-base

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



# Install ROS2
#FROM cuda-cudnn-base AS ros2-base
FROM ubuntu-base AS ros2-base

RUN sudo apt update && \
    sudo apt install -y --no-install-recommends curl gnupg2 lsb-release python3-pip vim wget build-essential ca-certificates

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN sudo apt update && sudo apt upgrade -y && \
    sudo apt install -y --no-install-recommends ros-humble-desktop ros-dev-tools ros-humble-usb-cam xvfb &&\
    #DEBIAN_FRONTEND=noninteractive sudo apt install -y --no-install-recommends ros-humble-desktop ros-dev-tools &&\
    sudo rm -rf /var/lib/apt/lists/*

RUN bash /opt/ros/humble/setup.sh 

# Install Nvidia Container Toollit
FROM ros2-base AS cuda-ros2-base

# RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
#     curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - && \
#     curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# RUN sudo apt-get update && apt-get install -y --no-install-recommends nvidia-container-toolkit

# Configuring ROS2
FROM cuda-ros2-base AS cuda-ros2

ARG ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}

ENV ROS_DOMAIN_ID ${ROS_DOMAIN_ID}
ENV ROS_LOCALHOST_ONLY ${ROS_LOCALHOST_ONLY}

# Configuring python environment
FROM cuda-ros2 AS cuda-ros2-python-base

# Installing python
ARG USER_NAME=${USER_NAME}
ENV HOME /home/${USER_NAME}
ENV POETRY_VERSION 1.3.1
ENV POETRY_PATH ${HOME}
ENV PATH $PATH:$HOME/.poetry/bin:$HOME/.local/bin:$HOME/bin:$PATH
USER ${USER_NAME}
WORKDIR ${HOME}/app
RUN sudo chown -R ${USER_NAME} ${HOME}  && sudo chmod -R 777 ${HOME}


# WORKDIR ${HOME}/app
# RUN sudo chown -R ${USER_NAME} ${HOME}  && sudo chmod -R 777 ${HOME}
# USER ${USER_NAME}

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

# Configure python dependencies
FROM cuda-ros2-python-base AS yamamiya-pm-ros2

ARG USER_NAME=${USER_NAME}
ENV HOME /home/${USER_NAME}

COPY ./pyproject.toml ${HOME}/app 
# COPY ./poetry.lock ${HOME}/app

### install python dependencies
RUN poetry config virtualenvs.create false && \
    poetry config installer.parallel false && \
#    poetry install
# RUN poetry install
   poetry export -f requirements.txt --output requirements.txt --without-hashes && \
   pip3 install -r requirements.txt --user --no-deps



# configure the ros environment
### for WSL case
ARG ENDPOINT_URL=${ENDPOINT_URL}
ENV DISPLAY=${ENDPOINT_URL}:0
RUN export DISPLAY=${ENDPOINT_URL}:0
#RUN sudo chmod 777 /dev/video*

# COPY .  ${HOME}/app
#RUN sudo chown -R ${USER_NAME} ${HOME}  && sudo chmod -R 777 ${HOME}
#COPY ./ros_start_up.sh  ${HOME}/app/scripts
#RUN sudo chown -R ${USER_NAME} ${HOME}/app/scripts && sudo chmod -R 777 ${HOME}/app/scripts && \
#    sudo chmod 777 ${HOME}/app/scripts/ros_start_up.sh



# initialize ROS and build ROS components
# RUN sudo rosdep init && rosdep update && . /opt/ros/humble/setup.bash
# WORKDIR /app/robotic_trade
#RUN rosdep install -i --from-path src --rosdistro humble -y && \
#    colcon build --packages-select market_server &&  . install/setup.bash

CMD ["/bin/bash"]


