#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source $SCRIPT_DIR/common.sh

cd $SCRIPT_DIR/../

# need writing permissons to build the packages
chmod -R 777 .

# Check if the container already exists
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} already exists. Starting the container..."
    docker start -i ${CONTAINER_NAME}
    #docker exec -it ${CONTAINER_NAME} "$@"
else
    echo "Creating and running a new container ${CONTAINER_NAME}..."
    docker run -it --gpus all --name ${CONTAINER_NAME} \
    -v "${PWD}:/home/user/Quamba:rw" \
    -w /home/user/Quamba \
    --shm-size 64G \
    "${IMAGE_NAME}" $@
fi