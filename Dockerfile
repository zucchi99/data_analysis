FROM debian:stable-slim

# set working directory
WORKDIR /app

# change shell from sh to bash (needed for use source command)
SHELL ["/bin/bash", "-c"] 

# import files
#COPY .. ../app
#COPY data/           data/
COPY uppaal/         uppaal/
COPY src/            src/
COPY main.py         main.py
COPY parameters.json parameters.json
COPY real_data__uppaal_simulations__association.json real_data__uppaal_simulations__association.json
COPY .docker/        .docker/

# update system
RUN apt-get update

#RUN apt install python3.11.6
RUN apt-get -y install pip python3-full
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get install python3.11.6.

# create a virtual environment 
RUN python3 -m venv /opt/venv
# add the venv to the path
ENV PATH="/opt/venv/bin:$PATH"
# activate venv
RUN source /opt/venv/bin/activate 
# install the needed python packages using the venv
RUN pip install wheel==0.41.2
RUN python3 -m pip install -r .docker/requirements.txt
