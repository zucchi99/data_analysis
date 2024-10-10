#!/bin/bash

#cd '/mnt/c/Users/franc/OneDrive - Università degli Studi di Udine/uni/4_anno/tirocinio/data_analysis/' 

# to debug
set -x

### add data to volume
cd './data/' 
# tar -czvf  data.tar.gz concentrations/ from_uppaal/ from_sensors/

image_version=4
container_version=tmp

my_img=uf-analysis-image-v${image_version}
my_cnt=uf-analysis-container-v${container_version}
my_vol=uf-analysis-dataset

data_dir="/app/data"
tmp_dir="/app/tmp"

# start & run a detached container
docker run -dit --name=${my_cnt} \
    --mount  source=${my_vol},target=${data_dir} \
    --volume ./:${tmp_dir} \
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
docker exec ${my_cnt} /bin/bash -c "cd ${tmp_dir} && cp -r ./ ${data_dir} && cd ${data_dir} && ls -l"

#docker container rm `docker ps -q -l`