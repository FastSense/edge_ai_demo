FROM ros:melodic-ros-base

ENV ROSUSER="user" \
    REALSENSE_ROS_VERSION="2.2.21"

RUN adduser --home /home/$ROSUSER --shell /bin/bash --gecos '' --disabled-password $ROSUSER \
    && usermod -a -G root,sudo,tty,dialout $ROSUSER \
    && echo "$ROSUSER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$ROSUSER

RUN apt-get update \
    && apt-get install -y \
        bash-completion \
    software-properties-common \
    && sed -i 's/--no-generate //' /usr/share/bash-completion/completions/apt-get \
    && sed -i 's/--no-generate //' /usr/share/bash-completion/completions/apt-cache \
    && apt-add-repository universe \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y \
        vim \
        screen \
        tmux \
        net-tools \
        byobu \
        python-pip \
        python3-pip \
        zbar-tools \
        ranger \
        neovim \
    && rm -rf /var/lib/apt/lists/*

RUN pip2 install --upgrade pip \
  && pip2 install  \
      socketIO_client \
      parse 

RUN pip3 install --upgrade pip \
    && pip3 install \
        pyzbar \
        onnxruntime \
        pyrealsense2 \
        nnio \
        pyyaml \
        rospkg \
        Flask \
        Flask-Cors \
        PyYAML \
        python-socketio \
        werkzeug \
        requests \
        roslibpy \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y \
        python-catkin-tools \
        ros-melodic-rosserial-python \
        ros-melodic-teleop-twist-keyboard \
        ros-melodic-rgbd-launch \
        ros-melodic-libuvc-camera \
        ros-melodic-teb-local-planner \
        ros-melodic-gmapping \
        ros-melodic-move-base \
        ros-melodic-robot-localization \
        ros-melodic-xacro \
        ros-melodic-robot-state-publisher \
        ros-melodic-ddynamic-reconfigure \
        ros-melodic-diagnostic-updater \
        ros-melodic-cv-bridge \
        ros-melodic-image-transport \
        ros-melodic-tf \
        ros-melodic-geometry \
        ros-melodic-compressed-image-transport \
        ros-melodic-rosbridge-server  \
        ros-melodic-web-video-server  \
        ros-melodic-rospy-message-converter \
        ros-melodic-joint-state-publisher \
    && rm -rf /var/lib/apt/lists/*

USER $ROSUSER

RUN mkdir /home/$ROSUSER/catkin_ws/ \
    && mkdir /home/$ROSUSER/catkin_ws/src \
    && echo "source /home/${ROSUSER}/catkin_ws/devel/setup.bash" >> ~/.bashrc \
    && echo "source /home/${ROSUSER}/scripts/aliases.sh" >> ~/.bashrc

CMD ["/bin/bash"]
