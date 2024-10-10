#!/bin/bash

# to debug
set -x

# move to the location of the .docker file
# cd '/mnt/c/Users/franc/OneDrive - Università degli Studi di Udine/uni/4_anno/tirocinio/data_analysis/' 
# set current version
image_version=5
container_version=5

my_img=uf-analysis-image-v${image_version}
my_cnt=uf-analysis-container-v${image_version}.${container_version}
my_vol=uf-analysis-dataset

# build docker image
docker build ./ -t ${my_img}
# start & run a detached container
docker run -dit --name=${my_cnt} \
    --mount source=${my_vol},target=/app/data \
    --entrypoint /bin/bash \
    ${my_img}

### to start & run an exited container
# docker start -a `docker ps -q -l`
#-a attach to container
#-i interactive mode
#docker ps List containers
#-q list only container IDs
#-l list only last created container

### to enter in a running container
docker exec -it ${my_cnt} /bin/bash