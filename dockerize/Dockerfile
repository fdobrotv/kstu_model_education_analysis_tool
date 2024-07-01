# FROM jupyter/tensorflow-notebook
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Windows Render related
USER root
RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false -y && \ 
    apt-get install -y xvfb && \
    apt-get install -y python3-opengl 

# Optional, needed for some environments
RUN apt-get install -y cmake && \
    apt-get install -y zlib1g zlib1g-dev 

USER ${NB_USER}

# Removed 
RUN pip install \
        numpy==1.21.2 jupyterlab gym keras opencv stable-baselines3 gymnasium[atari] gymnasium[accept-rom-license] ipywidgets \
        pyvirtualdisplay 

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# RUN mkdir -p ~/miniconda3
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# RUN rm -rf ~/miniconda3/miniconda.sh
# RUN ~/miniconda3/bin/conda init bash
# RUN ~/miniconda3/bin/conda init zsh
# RUN export PATH="/home/${NB_USER}/miniconda/bin:$PATH"

# Needed for some environments
RUN conda install swig
# RUN pip install box2d-py atari_py pystan

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# RUN sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false
RUN apt-get install -y nvidia-container-toolkit

# Just for the convenience of testing
# COPY ./example/my_atari_learn.ipynb /home/${NB_USER}/my_atari_learn.ipynb
# COPY ./example/env_render.ipynb /home/${NB_USER}/env_render.ipynb
# COPY ./example/18_reinforcement_learning.ipynb /home/${NB_USER}/18_reinforcement_learning.ipynb

# ENTRYPOINT [ "jupyter-lab" ]

# COPY jupyter.requirements.txt /jupyter.requirements.txt
COPY setup.jupyter.sh /setup.jupyter.sh
# RUN python3 -m pip install --no-cache-dir -r /jupyter.requirements.txt -U
# RUN pip install jupyter
# RUN pip install matplotlib
# RUN pip install jupyter_http_over_ws
# RUN pip install jupytext
# RUN pip install tensorboard
# RUN pip install jupyter_contrib_nbextensions

# RUN /setup.jupyter.sh

# RUN pip install tensorflow
# RUN pip install tensorflow[and-cuda]
# RUN pip install opencv-python
# RUN pip install jupyterlab
# RUN pip install gym
# RUN pip install gymnasium[atari]
# RUN pip install ipywidgets
# RUN pip install pyvirtualdisplay 
# RUN pip install keras 
# RUN pip install stable-baselines3  
# RUN pip install gymnasium[accept-rom-license]
# RUN pip install opencv
# RUN pip install numpy==1.21.1
# RUN pip install pygame
# RUN pip install gym[classic_control]
# RUN pip install numpy
# RUN pip install scikit-learn
# RUN pip install keras-rl2
# RUN pip install stable-baselines3[mpi]
RUN conda install python=3.8 -y

RUN pip install box2d-py 
RUN pip install atari_py 
RUN pip install pystan
RUN pip install tensorflow==2.3.0
RUN pip install tensorflow[and-cuda]

RUN pip install jupyter
RUN pip install matplotlib
RUN pip install jupyter_http_over_ws
RUN pip install jupytext
RUN pip install tensorboard
RUN pip install jupyter_contrib_nbextensions

RUN /setup.jupyter.sh

RUN pip install opencv-python
RUN pip install jupyterlab
RUN pip install gym
RUN pip install gymnasium[atari]
RUN pip install ipywidgets
RUN pip install pyvirtualdisplay 
RUN pip install keras 
RUN pip install stable-baselines3  
RUN pip install gymnasium[accept-rom-license]


RUN pip install pygame

RUN pip install numpy
RUN pip install scikit-learn
RUN pip install keras-rl2
RUN pip install stable-baselines3[mpi]
RUN pip install protobuf==3.20.*

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY ./ROMS/ /home/${NB_USER}/ROMS

EXPOSE 8888
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]