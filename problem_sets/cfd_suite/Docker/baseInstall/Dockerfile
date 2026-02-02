FROM openfoam/openfoam4-graphical-apps
USER root
SHELL ["/bin/bash", "-c"]
#ENV TERM xterm
#ENTRYPOINT ["/bin/bash", "-c"]

RUN rm /entry.sh ; touch /entry.sh ; chmod +x /entry.sh

# Copy the current directory contents into the container at /app
COPY . /app

# Install any extra things we might need
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim \
        ssh \
        wget \
        git \
        bash \
	make \
	build-essential \
        software-properties-common \
	python3-setuptools \
	python3-pip ; \
        add-apt-repository ppa:deadsnakes/ppa ; apt update ; apt install -y --no-install-recommends python3.7 ; \
        rm -rf /var/lib/apt/lists/*

RUN cd /usr/local/bin ; ln -s /usr/bin/python3.7 python ; python3.7 -m pip install pip ; pip3 install --upgrade pip

ENV USER foam
ENV TERM xterm
