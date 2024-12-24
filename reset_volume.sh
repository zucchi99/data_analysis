#!/bin/bash

# to debug
set -x

# image, container and volume names
my_img=uf-analysis-image
my_cnt=uf-analysis-container-TEMP
my_vol=uf-analysis-dataset

# reset the volume
docker volume rm     ${my_vol}
docker volume create ${my_vol}

# specify the paths
# NB in the local data is assumed that only the raw data are present
local_data_dir='./data/' 
cnt_data_dir="/app/data"
cnt_tmp_dir="/app/tmp"

# create a temporary container
# copy the local data to /app/tmp
# mount the volume to /app/data
docker run -dit --name=${my_cnt} \
    --mount  source=${my_vol},target=${cnt_data_dir} \
    --volume ${local_data_dir}:${cnt_tmp_dir} \
    --entrypoint /bin/bash \
    ${my_img}

# copy the data from /app/tmp to /app/data (so into the volume)
docker exec ${my_cnt} /bin/bash -c "cd ${cnt_tmp_dir} && cp -r ./ ${cnt_data_dir} && cd ${cnt_data_dir} && ls -l"

# delete temporary container
docker stop ${my_cnt}
docker container rm ${my_cnt}

### to start & run an exited container
# docker start -a `docker ps -q -l`
#-a attach to container
#-i interactive mode
#docker ps List containers
#-q list only container IDs
#-l list only last created container
