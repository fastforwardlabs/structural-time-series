# python base image
# FROM python:3.6.9
# FROM python:3.7
# FROM python:3.6-slim
FROM wajdikh/fbprophet:latest

# set a working directory
WORKDIR /usr/src/app

# copy all project files
COPY . ./

# install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --user -r requirements.txt
# RUN pip3 install numpy
# RUN pip3 install pandas
# RUN pip3 install fbprophet
