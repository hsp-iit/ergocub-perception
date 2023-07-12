FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV USER=ergocub
ARG PASSWORD=ergocub

ARG CONDA_SCRIPT=Mambaforge-Linux-x86_64.sh
ARG CONDA_LINK=https://github.com/conda-forge/miniforge/releases/latest/download/${CONDA_SCRIPT}
ENV CONDA_MD5=aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651

ENV PYTHONDONTWRITEBYTECODE=true

RUN apt update \
    && apt install --no-install-recommends -y -qq sudo git \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup ${USER} \
    && useradd -ms /bin/bash ${USER} -g ${USER} \
    && echo "${USER}:${PASSWORD}" | chpasswd \
    && usermod -a -G sudo ${USER} \
    && sed -i.bak -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

# From here you can use sudo without password

# Installing mamba
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends wget bzip2 \
    && wget ${CONDA_LINK} \
    && bash ./${CONDA_SCRIPT} -b \
    && /home/${USER}/mambaforge/bin/mamba init bash \
    && sudo find /home/${USER}/mambaforge -follow -type f -name '*.a' -delete \
    && sudo find /home/${USER}/mambaforge -follow -type f -name '*.pyc' -delete \
    && /home/${USER}/mambaforge/bin/mamba clean -afy \
    && rm ${CONDA_SCRIPT}

# # Installing python packages
# Can't use env.yml because cuda install asks to accept agreement
RUN sudo ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime && \
    sudo apt update && DEBIAN_FRONTEND=noninteractive sudo apt install ffmpeg libsm6 libxext6 -y

RUN /home/${USER}/mambaforge/bin/mamba create -y -n ergocub \
    python=3.10 loguru pytorch torchvision pytorch-cuda=11.8 cuml=23.06 \
    cudatoolkit=11.8 einops tqdm opencv matplotlib pyquaternion vispy pysimplegui timm dgl yarp -c rapidsai -c nvidia -c pytorch -c robotology -c dglteam/label/cu118

RUN /home/${USER}/mambaforge/envs/ergocub/bin/pip install nvidia-pyindex nvidia-tensorrt \
    polygraphy pycuda open3d omegaconf mediapipe pyrealsense2

# # After this you will need the password to run sudo
RUN sudo mv /etc/sudoers.bak /etc/sudoers
RUN echo "mamba activate ergocub" >> /home/${USER}/.bashrc

CMD ["bash"]
