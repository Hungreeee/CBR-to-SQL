FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /cv-poc

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY ./app ./app

# Expose the port the app runs on
EXPOSE 8000

# Set the default command to run the application
CMD ["python", "app/run.py"]
