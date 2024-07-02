#!/bin/bash
source common.sh

# sudo apt-get update
# sudo apt-get install -y nvidia-docker2
# sudo systemctl restart docker
docker build -f Dockerfile -t "${IMAGE_NAME}" .
