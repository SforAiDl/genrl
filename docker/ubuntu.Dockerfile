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

COPY ./requirements.txt requirements.txt

RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /usr/local/miniconda \
    && hash -r \
    && /usr/local/miniconda/bin/conda config --set always_yes yes --set changeps1 no \
    && /usr/local/miniconda/bin/conda update -q conda \
    #&& /usr/local/miniconda/bin/conda install -q conda-env \
    && /usr/local/miniconda/bin/conda create -q -n genrl python=3.6 wheel virtualenv pytest readme_renderer pandas cmake nomkl numpy boost py-boost scipy scikit-learn six joblib \ 
    && /usr/local/miniconda/bin/conda clean -a \
    && /usr/local/miniconda/bin/conda init \
    && rm miniconda.sh

ENV PATH="/usr/local/miniconda/bin:${PATH}"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN /usr/local/miniconda/envs/genrl/bin/python -m pip install -r requirements.txt
    && /usr/local/miniconda/envs/genrl/bin/python -m pip install black isort==4.3.2 flake8 codecov pytest-cov
