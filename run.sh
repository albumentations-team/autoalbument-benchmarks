#!/bin/bash

DATA_DIR=$1
OUTPUTS_DIR=$2

docker build -t autoalbument_benchmarks:latest -f ./docker/Dockerfile .

docker run --gpus all --rm -it --shm-size=2g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest cifar10_baseline
docker run --gpus all --rm -it --shm-size=2g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest cifar10_autoalbument

docker run --gpus all --rm -it --shm-size=2g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest svhn_baseline
docker run --gpus all --rm -it --shm-size=2g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest svhn_autoalbument

docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest imagenet_baseline
docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest imagenet_autoalbument

docker run --gpus all --rm -it --shm-size=8g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest pascal_voc_baseline
docker run --gpus all --rm -it --shm-size=8g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest pascal_voc_autoalbument

docker run --gpus all --rm -it --shm-size=8g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest cityscapes_baseline
docker run --gpus all --rm -it --shm-size=8g --ulimit memlock=-1 -v "${DATA_DIR}":/data -v "${OUTPUTS_DIR}":/outputs autoalbument_benchmarks:latest cityscapes_autoalbument
