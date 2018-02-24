# params for install of tensorflow lib
INCLUDE_DIR=./include
SRC_DIR=./src/
LIB_DIR=./lib/
OUT_DIR=./bin/

# tensorflow lib location
TF_TYPE=cpu

ifeq ($(OS),Windows_NT)
	OS_NAME=windows
	OS_FLAG=_WIN32
	EXTN=zip
	UNPACK_CMD=unzip
else
	OS_NAME=linux
	OS_FLAG=__linux__
	EXTN=tar.gz
	UNPACK_CMD=tar -xvf
endif

TF_LIB_NAME=libtensorflow-$(TF_TYPE)-$(OS_NAME)-x86_64-1.5.0.$(EXTN)
TF_LIB_LOC="https://storage.googleapis.com/tensorflow/libtensorflow/$(TF_LIB_NAME)"

CC=gcc
CFLAGS=-I$(INCLUDE_DIR) -D$(OS_FLAG)
LDFLAGS=-L$(LIB_DIR) -ltensorflow


#g++ main.cpp -o main.out -I./include -L./lib/libtensorflow/ -libtensorflow

$(OUT_DIR):
	mkdir $(OUT_DIR)

tf_hello_world: $(SRC_DIR)tf_hello_world.c $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(OUT_DIR)$@ $< $(LDFLAGS)

tf_load_graph: $(SRC_DIR)tf_load_graph.c $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(OUT_DIR)$@ $< $(LDFLAGS)

tf_graph_info: $(SRC_DIR)tf_graph_info.c $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(OUT_DIR)$@ $< $(LDFLAGS)

tf_session: $(SRC_DIR)tf_session.c $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(OUT_DIR)$@ $< $(LDFLAGS)

### DOWNLOAD AND UNPACK TENSORFLOW C API LIB ###
$(TF_LIB_NAME):
	curl $(TF_LIB_LOC) > $(TF_LIB_NAME)

unpack: $(TF_LIB_NAME)
	$(UNPACK_CMD) $(TF_LIB_NAME)

fetch: unpack

### MAKE A LIB FOR THE DLL WIN32 ###
tensorflow.def: $(OUT_DIR)tensorflow.dll
	echo EXPORTS > $(LIB_DIR)tensorflow.def
	dumpbin /EXPORTS $(OUT_DIR)tensorflow.dll | tail -n 204 | head -194 | awk '{ print $$4 }' >> $(LIB_DIR)tensorflow.def

tensorflow.lib: tensorflow.def
	 lib /def:$(LIB_DIR)$< /OUT:$(LIB_DIR)$@ /MACHINE:x64

### ALL ###
all: tf_hello_world tf_load_graph tf_graph_info tf_session
