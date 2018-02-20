# params for install of tensorflow lib
INCLUDE_DIR=./include
LIB_DIR=./lib/libtensorflow

# tensorflow lib location
TF_TYPE=cpu
OS=linux
TF_LIB_NAME=libtensorflow-$(TF_TYPE)-$(OS)-x86_64-1.5.0.tar.gz
TF_LIB_LOC="https://storage.googleapis.com/tensorflow/libtensorflow/$(TF_LIB_NAME)"

CC=gcc
CFLAGS=-I$(INCLUDE_DIR)
LDFLAGS=-L$(LIB_DIR) -ltensorflow


#g++ main.cpp -o main.out -I./include -L./lib/libtensorflow/ -libtensorflow

tf_hello_world: tf_hello_world.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tf_load_graph: tf_load_graph.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tf_graph_info: tf_graph_info.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tf_session: tf_session.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm *.o

fetch:
	curl $(TF_LIB_LOC) > $(TF_LIB_NAME)
	tar -xvf $(TF_LIB_NAME)
	rm $(TF_LIB_NAME)
