# Use an official Python runtime as a parent image   
FROM python:3.11-slim-buster

# Update and install libs
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# update pip
RUN pip install --upgrade pip
# Set default timeout for pip
RUN pip install --default-timeout=900 future
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


# Make port 8100 available to the world outside this container
EXPOSE 8100

# Run object_detector.py when the container launches
CMD ["python", "api.py"]