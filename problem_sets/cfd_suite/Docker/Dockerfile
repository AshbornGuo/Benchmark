FROM frehbach/cfd-test-problem-suite_base

#SHELL ["/bin/bash"]
ENTRYPOINT ["/bin/bash"]

ENV USER foam
ENV TERM xterm

RUN chown $(whoami) ~/ ; chown $(whoami) /app ; cd /app ; pip3 install --trusted-host pypi.python.org -r requirements.txt 

RUN source /opt/openfoam4/etc/bashrc ; cd ../ ; mkdir cfMesh; wget https://sourceforge.net/projects/cfmesh/files/v1.1.2/cfMesh-v1.1.2.tgz ; \
    tar zxvf cfMesh-v1.1.2.tgz ; cd cfMesh-v1.1.2/ ; chmod u+x ./Allwmake ; ./Allwmake 

## Install cfd_Test suite
RUN cd ~/ ; git clone https://frehbach@bitbucket.org/arahat/cfd-test-problem-suite.git 

## Only while this branch is being developed
#RUN cd cfd-test-problem-suite ; git checkout dockerSetup

RUN source /opt/openfoam4/etc/bashrc ; \
    cd ~/cfd-test-problem-suite/Exeter_CFD_Problems/data/PitzDaily/solvers/ ; ./Allmake ; \
    cd ~/cfd-test-problem-suite/Exeter_CFD_Problems/data/KaplanDuct/solvers/ ; ./Allmake ; \
    cd ~/cfd-test-problem-suite/Exeter_CFD_Problems/data/HeatExchanger/solvers/ ; ./Allmake

WORKDIR cfd-test-problem-suite

# Define environment variable
ENV USER foam
ENV TERM xterm

COPY dockerCall.py .
COPY dockerCall.sh .
COPY SnappyHexOptimise.py ./Exeter_CFD_Problems/data/
COPY ESP ./Exeter_CFD_Problems/ESP

#RUN cd /opt/openfoam4 ; rm -r tutorials ; rm -r wmake ; rm -r src ; rm -r doc
#RUN apt-get -y remove emacs gedit gedit-plugins gnuplot gnuplot-x11 gnuplot-doc meld vim ; apt-get -y auto-remove ; apt-get -y autoclean
