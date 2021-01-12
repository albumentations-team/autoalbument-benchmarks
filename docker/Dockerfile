FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /autoalbument_benchmarks
COPY requirements.txt /autoalbument_benchmarks/requirements.txt
RUN pip install --no-cache-dir -r /autoalbument_benchmarks/requirements.txt
COPY docker/entrypoint.sh entrypoint.sh
COPY . .

ENV DATA_DIR=/data
ENV OUTPUTS_DIR=/outputs

ENTRYPOINT ["/autoalbument_benchmarks/entrypoint.sh"]
