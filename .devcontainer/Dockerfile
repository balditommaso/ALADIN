FROM    ubuntu:focal
SHELL   ["/bin/bash", "-c"]
# Set non-interactive mode globally
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.9 \
        python3.9-venv \
        python3-pip \
        build-essential \
        git \
        libftdi-dev \
        libftdi1 \
        doxygen \
        libsdl2-dev \
        curl \
        cmake \
        libusb-1.0-0-dev \
        gtkwave \
        libsndfile1-dev \
        rsync \
        autoconf \
        automake \
        texinfo \
        libtool \
        pkg-config \
        libsdl2-ttf-dev \
        wget \
        unzip \
        sed && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1


#RUN     stat /etc/udev/
#RUN     cp /usr/share/gap8-openocd/openocd/contrib/60-openocd.rules /etc/udev/rules.d
#RUN     udevadm control --reload-rules && sudo udevadm trigger
#RUN     usermod -a -G dialout ubuntu
# GAP-SDK & TOOLCHAIN INSTALLATION
RUN     python3 -m venv /dory_env && \
        source /dory_env/bin/activate && \
        git clone https://github.com/GreenWaves-Technologies/gap8_openocd.git && \
        cd gap8_openocd && \
       ./bootstrap && \
        ./configure --program-prefix=gap8- --prefix=/usr --datarootdir=/usr/share/gap8-openocd && \
        make -j && \
        make -j install
RUN     source /dory_env/bin/activate && \
		cd / && \
        git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git && \
        cd /gap_riscv_toolchain_ubuntu_18 && \
        ./install.sh /usr/lib/gap_riscv_toolchain && \
        git clone https://github.com/GreenWaves-Technologies/gap_sdk/ && \
        cd gap_sdk && \        
        git checkout a3dedd5cd8a680a88d2dca2ab7a4ae65cebf4c8d && \
        python3 -m pip install wheel && \
        python3 -m pip install scons && \
        python3 -m pip install Cython && \
        python3 -m pip install --upgrade pip setuptools wheel && \
        python3 -m pip install setuptools_rust && \
        python3 -m pip install -r requirements.txt
RUN     source /dory_env/bin/activate && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk && \
        source sourceme.sh && \
        make minimal && \
        make gvsoc && \
# PULP-SDK INSTALLATION
        cd / && \
        git clone https://github.com/pulp-platform/pulp-sdk.git && \
        cd pulp-sdk && \
        git checkout 1f6a59e9b3def1585cd726872ea93b134a46faa2 && \
        source configs/pulp-open-nn.sh && \
        make all && \
# PULP-NN TOOLCHAIN DOWNLOAD
        cd / && \
        wget https://retis.santannapisa.it/~tbaldi/download/riscv-nn-toolchain.zip && \
        unzip riscv-nn-toolchain
# FINALIZE PYTHON VENV
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/
RUN     source /dory_env/bin/activate && \
        python3 -m pip install python-dev-tools --upgrade && \
        python3 -m pip install numpy && \
        python3 -m pip install onnx onnxscript onnxoptimizer && \
        python3 -m pip install future-fstrings && \
        python3 -m pip install pandas && \
        python3 -m pip install qonnx ortools && \
        python3 -m pip install mako && \
        python3 -m pip install pytest && \
        python3 -m pip install Pygments && \
        python3 -m pip install MarkupSafe && \
        python3 -m pip install prettytable && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk/ && \
        python3 -m pip uninstall -y torch torchvision && \
        python3 -m pip install torch torchvision && \
        python3 -m pip install brevitas pytorch-lightning tensorboard && \
        python3 -m pip install argcomplete pyelftools && \
        python3 -m pip install matplotlib

# VIVADO FLOW
RUN echo '\
if [ -z "\$VIVADO_HOME\" ]; then\n\
        echo \"[Vivado] WARNING: VIVADO_HOME not set. FPGA flow disabled.\" >&2\n\
        return 0\n\
fi\n\
\n\
if [ ! -d \"$VIVADO_HOME\" ]; then\n\
        echo \"[Vivado] ERROR: VIVADO_HOME=$VIVADO_HOME does not exist\" >&2\n\
        return 0\n\
fi\n\
\n\
if [ ! -f \"$VIVADO_HOME/settings64.sh\" ]; then\n\
        echo \"[Vivado] ERROR: Invalid Vivado install at $VIVADO_HOME\" >&2\n\
        return 0\n\
fi\n\
\n\
source \"$VIVADO_HOME/settings64.sh\"\n\
export PATH=\"$VIVADO_HOME/bin:$PATH\"\n\
\n\
if [ -n \"$VIVADO_VERSION\" ]; then\n\
        echo \"[Vivado] Enabled: version $VIVADO_VERSION\"\n\
else\n\
        echo \"[Vivado] Enabled at $VIVADO_HOME\"\n\
fi' > /etc/profile.d/vivado.sh

RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    libtinfo5 libncurses5 libx11-6 libxext6 libxft2 libxrender1 libxrandr2 libglib2.0-0


ARG USERNAME=dory
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
