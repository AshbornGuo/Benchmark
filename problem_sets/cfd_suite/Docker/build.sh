docker build --tag="frehbach/cfd-test-problem-suite_base" baseInstall/ $1
docker build --tag="frehbach/cfd-test-problem-suite" . $1
#docker-squash -t frehbach/cfd-test-problem-suite frehbach/cfd-test-problem-suite
