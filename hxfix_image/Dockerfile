FROM frehbach/cfd-test-problem-suite

WORKDIR /home/openfoam/cfd-test-problem-suite

COPY patch_init.py /tmp/patch_init.py
COPY patch_dockercall.py /tmp/patch_dockercall.py
COPY patch_shell.py /tmp/patch_shell.py

RUN python3 /tmp/patch_init.py
RUN python3 /tmp/patch_dockercall.py
RUN python3 /tmp/patch_shell.py


COPY patch_shell_verbose.py /tmp/patch_shell_verbose.py
RUN python3 /tmp/patch_shell_verbose.py