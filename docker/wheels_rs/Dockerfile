ARG PROJECT_NAME

FROM ubuntu:18.04 AS builder

ENV REALSENSE_LIBRARY="v2.41.0"

RUN apt-get update \
    && apt-get install -y \
        git \
        libssl-dev \
        libusb-1.0-0-dev \
        pkg-config \
        libgtk-3-dev \
        cmake

# https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md#building-librealsense2-sdk
# https://github.com/IntelRealSense/librealsense/issues/7099
RUN mkdir prebuild \
    && cd prebuild \
    && git clone https://github.com/IntelRealSense/librealsense.git \
    && cd librealsense \
    && git checkout -b tags/${REALSENSE_LIBRARY} \
    && mkdir build \
    && cd build \
    && cmake .. -DFORCE_RSUSB_BACKEND=ON -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false -DBUILD_WITH_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release \
    && make \
    && make install DESTDIR=/rsinstall

FROM ${PROJECT_NAME}_base:latest

USER root

COPY --from=builder /rsinstall/usr/ /usr/

CMD ["/bin/bash"]
