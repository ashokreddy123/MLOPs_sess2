# our base image
FROM python:3.7.10


# set working directory inside the image
WORKDIR /app

# copy our requirements
COPY requirements.txt requirements.txt


# install dependencies
RUN pip3 install -r requirements.txt


#ENTRYPOINT ["/bin/bash"]

#CMD ['bash']