#from alpine/make as build
#
#run apk update && \
#    apk add --no-cache \
#      build-base \
#      gcc

from debian:stable-slim as build

run apt-get update && \
    apt-get install --no-install-recommends -y \
      build-essential \
      ca-certificates \
      wget

run mkdir /build
workdir /build

ARG TF_TYPE=cpu
ARG TF_VERSION=2.6.0

# install tf lib
run wget -q https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-$TF_TYPE-linux-x86_64-$TF_VERSION.tar.gz
run tar -xvf ./libtensorflow-$TF_TYPE-linux-x86_64-$TF_VERSION.tar.gz

copy ./src ./src
copy ./include/tf_utils.h ./include/
copy ./Makefile .

run make all

# build a tar file to preserve the symlinks, otherwise we duplicate
# tensorflow.so (~300Mb) three times.
run tar cfz deploy.tar.gz ./bin/* ./lib/*


from debian:stable-slim as deploy

copy --from=build /build/deploy.tar.gz /
run tar xvf deploy.tar.gz
