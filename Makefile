# params for install of tensorflow lib
INCLUDE_DIR=./include
LIB_DIR=./lib/libtensorflow

# tensorflow lib location
TF_TYPE=cpu

ifeq ($(OS),Windows_NT)
	OS_NAME=windows
	EXTN=zip
	UNPACK_CMD=unzip
else
	OS_NAME=linux
	EXTN=tar.gz
	UNPACK_CMD=tar -xvf
endif

TF_LIB_NAME=libtensorflow-$(TF_TYPE)-$(OS_NAME)-x86_64-1.5.0.$(EXTN)
TF_LIB_LOC="https://storage.googleapis.com/tensorflow/libtensorflow/$(TF_LIB_NAME)"

CC=gcc
CFLAGS=-I$(INCLUDE_DIR)
LDFLAGS=-L$(LIB_DIR) -ltensorflow


#g++ main.cpp -o main.out -I./include -L./lib/libtensorflow/ -libtensorflow

tf_hello_world: src/tf_hello_world.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tf_load_graph: src/tf_load_graph.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tf_graph_info: src/tf_graph_info.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tf_session: src/tf_session.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

### DOWNLOAD AND UNPACK TENSORFLOW C API LIB ###
$(TF_LIB_NAME):
	curl $(TF_LIB_LOC) > $(TF_LIB_NAME)

unpack: $(TF_LIB_NAME)
	$(UNPACK_CMD) $(TF_LIB_NAME)

fetch: unpack

### ALL ###
all: fetch tf_hello_world tf_load_graph tf_graph_info tf_session
