ARG PROJECT_NAME

FROM ${PROJECT_NAME}_rs_ov:latest

USER root

# >>>> Install some basic utilities
RUN apt-get update \
    && apt-get install -y \
        gnupg \
        kmod \
        pciutils \
        unzip \
        wget \
        # OpenCV dependencies
        ffmpeg \
        libsm6 \
        libxext6
# <<<< Install some basic utilities


# >>>> Install python and pip
# Install python
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && pip3 install -U pip \
    && pip3 install \
        opencv-python \
        nnio
# <<<< Install python and pip


# >>>> Install google coral driver
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y \
        gasket-dkms \
        libedgetpu1-std \
    && wget https://github.com/google-coral/edgetpu/files/5052663/edgetpu-2.14.1-py3-none-any.zip \
    && unzip edgetpu-2.14.1-py3-none-any.zip \
    && pip3 install edgetpu-2.14.1-py3-none-any.whl \
    && pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
# <<<< Install tflite_runtime

USER user

WORKDIR /home/user
RUN sudo groupadd apex \
    && sudo adduser user apex
# <<<< Create user
