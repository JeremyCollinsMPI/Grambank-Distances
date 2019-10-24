FROM tensorflow/tensorflow:2.0.0
RUN pip install pandas geocoder scipy
WORKDIR /directory
