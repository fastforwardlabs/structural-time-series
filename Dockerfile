# use existing fbprophet image because difficulty 
# with base python and fbprophet...
FROM wajdikh/fbprophet:latest

# set a working directory
WORKDIR /usr/src/app

# copy all project files
COPY . ./

# install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --user -r kfp/requirements.txt

# install the sts module
RUN pip3 install -e .