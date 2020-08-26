FROM ubuntu:latest

# Install baseline required tools
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
    && apt-get update && \
    apt install -y --no-install-recommends \
    # C++
    build-essential g++ gcc git make wget \
    # Python
    python-setuptools python-dev \
    sudo \
    libgtk2.0-dev \
    && apt-get install -y --reinstall ca-certificates \
    # Do this cleanup every time to ensure minimal layer sizes
    # TODO: Turn this into a script
    && apt-get clean autoclean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}


RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /usr/local/miniconda \
    && hash -r \
    && /usr/local/miniconda/bin/conda config --set always_yes yes --set changeps1 no \
    && /usr/local/miniconda/bin/conda update -q conda \
    #&& /usr/local/miniconda/bin/conda install -q conda-env \
    && /usr/local/miniconda/bin/conda create -q -n test-python36 python=3.6 wheel virtualenv pytest readme_renderer pandas cmake nomkl numpy boost py-boost scipy scikit-learn six joblib \ 
    # atari-py==0.2.6 box2d-py==2.3.8 certifi==2019.11.28 cloudpickle==1.3.0 future==0.18.2 gym==0.17.1 numpy==1.18.2 opencv-python==4.2.0.34 Pillow==7.0.0 pyglet==1.5.0 scipy==1.4.1 six==1.14.0 matplotlib==3.2.1 pytest==5.4.1 torch==1.4.0 torchvision==0.5.0 tensorflow-tensorboard==1.5.1 tensorboard==1.15.0 pre-commit==2.4.0 importlib-resources==1.0.1 setuptools==41.0.0 black isort==4.3.2 flake8 codecov pytest-cov\
    && /usr/local/miniconda/bin/conda clean -a \
    # init is needed to ensure that the environment is properly set up for "source activate"
    && /usr/local/miniconda/bin/conda init \
    && rm miniconda.sh

ENV PATH="/usr/local/miniconda/bin:${PATH}"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN /usr/local/miniconda/envs/test-python36/bin/python -m pip install atari-py==0.2.6 box2d-py==2.3.8 certifi==2019.11.28 cloudpickle==1.3.0 future==0.18.2 gym==0.17.1 numpy==1.18.2 opencv-python==4.2.0.34 Pillow==7.0.0 pyglet==1.5.0 scipy==1.4.1 six==1.14.0 matplotlib==3.2.1 pytest==5.4.1 torch==1.4.0 torchvision==0.5.0 tensorflow-tensorboard==1.5.1 tensorboard==1.15.0 pre-commit==2.4.0 importlib-resources==1.0.1 setuptools==41.0.0 \
    # CI requirements
    && /usr/local/miniconda/envs/test-python36/bin/python -m pip install black isort==4.3.2 flake8 codecov pytest-cov
