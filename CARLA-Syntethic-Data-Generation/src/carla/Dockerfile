FROM carlasim/carla:0.9.15

USER root
RUN apt-get update && apt-get install -y wget
RUN wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
RUN mv AdditionalMaps_0.9.15.tar.gz /home/carla/Import

WORKDIR /home/carla/Import
RUN tar -xf AdditionalMaps_0.9.15.tar.gz

WORKDIR /home/carla
RUN ./ImportAssets.sh
RUN rm -f /home/carla/Import/AdditionalMaps_0.9.15.tar.gz
USER carla