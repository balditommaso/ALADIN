FROM    ubuntu:focal
SHELL   ["/bin/bash", "-c"]
# Set non-interactive mode globally
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common sudo gnupg curl ca-certificates && \
    # Fetch the deadsnakes PPA signing key over plain HTTPS (port 443)
    # instead of the keyserver protocol (port 11371), which frequently
    # times out inside Docker builds — this is the standard, reliable fix.
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        -o /tmp/deadsnakes.asc && \
    gpg --dearmor -o /usr/share/keyrings/deadsnakes-archive-keyring.gpg /tmp/deadsnakes.asc && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes-archive-keyring.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu focal main" \
        > /etc/apt/sources.list.d/deadsnakes.list && \
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


RUN chown -R $USER_UID:$USER_GID /dory_env

USER $USERNAME

# Auto-activate the venv and GAP SDK env vars in every interactive shell
RUN echo "source /dory_env/bin/activate" >> /home/$USERNAME/.bashrc && \
    echo "source /gap_riscv_toolchain_ubuntu_18/gap_sdk/sourceme.sh" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]