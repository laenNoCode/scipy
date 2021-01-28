ARG BASE_CONTAINER=trallard/scipy:20210127-feature-Docker-b7f5aac1ce8009af05e9011152180cc3112cf13f
FROM ${BASE_CONTAINER}

# -----------------------------------------------------------------------------
# ---- OS dependencies ----
# hadolint ignore=DL3008
RUN apt-get update && \ 
    apt-get install -yq --no-install-recommends \
    zip \
    unzip \
    bash-completion \
    build-essential \
    htop \
    jq \
    less \
    locales \
    man-db \
    sudo \
    time \
    multitail \
    lsof \
    ssl-cert \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

RUN add-apt-repository -y ppa:git-core/ppa \
    && apt-get install -yq git \
    && rm -rf /var/lib/apt/lists/* \
    conda activate scipydev && \
    conda install conda-build -y

# -----------------------------------------------------------------------------
# ---- Gitpod users ----
ENV LANG=en_US.UTF-8 \
    HOME=/home/gitpod

# '-l': see https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
RUN useradd -l -u 33333 -G sudo -md "${HOME}" -s /bin/bash -p gitpod gitpod && \
    # passwordless sudo for users in the 'sudo' group
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers && \
    # vanity custom bash prompt
    { echo && echo "PS1='\[\e]0;\u \w\a\]\[\033[01;36m\]\u\[\033[m\] > \[\033[38;5;141m\]\w\[\033[m\] \\$'" ; } >> .bashrc

WORKDIR $HOME

USER gitpod
