#!/bin/bash

# to debug
set -x

# image, container and volume names
my_img=uf-analysis-image
my_cnt=uf-analysis-container
my_vol=uf-analysis-dataset

# create the docker volume
docker volume create ${my_vol}

# build docker image
docker build ./ -t ${my_img}

# fill the volume
/bin/bash ./reset_volume.sh

# start & run a detached container
docker run -dit --name=${my_cnt} \
    --mount source=${my_vol},target=/app/data \
    --entrypoint /bin/bash \
    ${my_img}

# enter in the running container
docker exec -it ${my_cnt} /bin/bash

# export container and volume data
# docker cp <container>:<cont_path> <local_path>
