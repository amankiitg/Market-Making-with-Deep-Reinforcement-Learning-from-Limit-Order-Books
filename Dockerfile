# Faithful Linux x86_64 stack for reproducing the paper's original environment.
#
# Build and run (the base image is pinned to amd64 because the tensorflow 2.6 wheels only
# exist for x86_64 and require AVX, which Docker Desktop on Apple Silicon does not emulate
# for this code path: expect to use a cloud Linux box instead of Docker Desktop on the Mac):
#
#   docker build --platform linux/amd64 -t mm-lob:tf26 .
#   docker run --rm -it --platform linux/amd64 -v "$PWD:/workspace" mm-lob:tf26 bash
#   # inside the container
#   python -m data.adapter --synthetic --rows 4000
#   python pretrain.py --days 20191101
#   python main.py --train-days 20191101 --test-days 20191101 --save=True
#
# On an Apple Silicon host you can still build the image (it is amd64, emulated) but
# TensorFlow 2.6 will abort with "Illegal instruction" because QEMU does not implement the
# AVX instructions the wheel uses. Use this image on an x86_64 Linux machine.

FROM --platform=linux/amd64 python:3.8-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Pins from the repository's conda_setup.yaml, with the TensorFlow 2.6 companion packages
# corrected to the 2.6.x line (the original file pinned tensorflow==2.6.0 together with
# tensorflow-estimator==2.10.0 and tensorboard==2.10.1, which is an inconsistent set).
RUN pip install --upgrade "pip<24" "setuptools<60" wheel \
 && pip install \
      tensorflow==2.6.0 \
      tensorflow-estimator==2.6.0 \
      tensorboard==2.6.0 \
      keras==2.6.0 \
      numpy==1.19.5 \
      h5py==3.1.0 \
      pandas==1.3.5 \
      scipy==1.9.1 \
      scikit-learn==1.1.2 \
      tqdm==4.64.1 \
      protobuf==3.19.6 \
      msgpack==1.0.4 \
      msgpack-numpy==0.4.8 \
      pyrallis==0.3.1 \
      typing-inspect==0.9.0 \
      PyYAML==6.0 \
      pytest==7.1.3

# tensorforce 0.6.5 must be installed without dependencies: its metadata pins
# tensorflow==2.6.0 (satisfied above) together with numpy==1.19.5 and h5py~=3.1.0, and if
# pip is allowed to resolve it will fight the tensorflow 2.6 constraints. The runtime
# package set above is the one it actually needs (tensorflow, numpy, tqdm, h5py, msgpack,
# msgpack-numpy).
RUN pip install --no-deps tensorforce==0.6.5

# Sanity check, fails the build early if the stack is broken.
RUN python -c "import tensorflow, tensorforce, pandas, numpy; \
print('tensorflow', tensorflow.__version__, 'tensorforce', tensorforce.__version__, \
'pandas', pandas.__version__, 'numpy', numpy.__version__)"

COPY . /workspace

# compat.py is a no-op on tensorflow < 2.11, so it protects both this image and the Mac venv.
CMD ["python", "-c", "from compat import apply_tensorforce_tf_compat; apply_tensorforce_tf_compat(); import tensorforce, tensorflow; print('stack ready', tensorflow.__version__, tensorforce.__version__)"]
