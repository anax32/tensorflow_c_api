# Tensorflow C API

Simple examples which wrap the tensorflow C API for inference/introspection etc

## examples

+ test
```bash
docker build -t test .
docker run -it --rm test tf_hello_world
```

+ test old tensorflow version
```bash
docker build --build-arg TF_VERSION=2.4.0 -t test .
docker run --rm -it test tf_hello_world
```

+ inference


## refs

+ tensorflow library
    + https://www.tensorflow.org/install/lang_c
