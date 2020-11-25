FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN python -m pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN python -m pip  install -r requirements.txt
