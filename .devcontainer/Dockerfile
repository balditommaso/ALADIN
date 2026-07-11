FROM    ubuntu:focal
SHELL   ["/bin/bash", "-c"]
# Set non-interactive mode globally
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 is built from source rather than via apt/deadsnakes: deadsnakes
# has dropped all package builds for Ubuntu Focal (20.04) since it reached
# EOL in April 2025 — `apt-get install python3.10` there now fails with
# "no installation candidate" regardless of PPA/key setup. Building from
# source has no such dependency and works on any base.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        sudo \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev \
        libftdi-dev \
        libftdi1 \
        doxygen \
        libsdl2-dev \
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
        unzip \
        sed && \
    cd /usr/src && \
    curl -fsSL -O https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz && \
    tar -xf Python-3.10.15.tgz && \
    cd Python-3.10.15 && \
    ./configure --enable-optimizations --enable-shared --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    ldconfig && \
    cd / && rm -rf /usr/src/Python-3.10.15* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10

# GAP-SDK & TOOLCHAIN INSTALLATION
# NOTE: this layer is expensive (builds openocd, riscv toolchain, gap_sdk,
# pulp-sdk from source). It only reruns if one of these RUN commands changes,
# so we deliberately keep it free of anything you'll want to tweak often
# (like Python package lists) — those live below in their own layer.
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
        git checkout a3dedd5cd8a680a88d2dca2ab7a4ae65cebf4c8d

# BUILD-TIME PYTHON DEPS — NOT part of the modular "edit anytime"
# requirements.txt below. `make minimal`/`make gvsoc` shell out to scons
# during the build itself, so scons (and friends) must exist in the venv
# *before* that step runs. This list is small and essentially never
# changes, so pinning it here (rather than in the late layer) is the right
# tradeoff — it only costs a rebuild if you ever touch it, which should be
# close to never.
COPY requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    source /dory_env/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements-build.txt

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
        wget https://iis-nextcloud.ee.ethz.ch/s/aYESyR5W9FrHgYa/download/riscv-nn-toolchain.zip && \
        unzip riscv-nn-toolchain

# --------------------------------------------------------------------------
# PYTHON DEPENDENCIES — kept as its own layer, driven by requirements.txt.
# Editing requirements.txt only invalidates *this* layer and everything
# below it — the expensive toolchain build above stays cached.
# --------------------------------------------------------------------------
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/

COPY requirements.txt /tmp/requirements.txt

# Uses BuildKit's pip cache mount so re-downloaded wheels are cached across
# builds even when this layer itself has to rerun. Build with:
#   DOCKER_BUILDKIT=1 docker build -t dory .
RUN --mount=type=cache,target=/root/.cache/pip \
    source /dory_env/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip uninstall -y torch torchvision || true && \
    python3 -m pip install -r /tmp/requirements.txt

# --------------------------------------------------------------------------
# USER SETUP
# --------------------------------------------------------------------------
ARG USERNAME=dory
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# /dory_env was created by root during the build steps above. The dev
# container's postCreateCommand/postStartCommand run as $USERNAME, so it
# needs write access to the venv (pip install, etc.) — hand it over here,
# before switching USER below.
RUN chown -R $USER_UID:$USER_GID /dory_env

USER $USERNAME

# Auto-activate the venv in every shell — this is non-interactive/safe.
#
# We deliberately do NOT auto-source gap_sdk/sourceme.sh here: it prompts
# interactively (`read`) for a board target. VS Code's Dev Containers
# extension also spawns non-interactive probe shells that source .bashrc
# (to capture PATH, etc.) — when one of those hits an interactive `read`,
# it consumes whatever's next in that input stream instead of real
# keyboard input, producing garbled errors. Instead, expose it as an
# alias you run explicitly, once, in a real terminal when you need it.
RUN echo "source /dory_env/bin/activate" >> /home/$USERNAME/.bashrc && \
    echo "alias gap-init='source /gap_riscv_toolchain_ubuntu_18/gap_sdk/sourceme.sh'" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]