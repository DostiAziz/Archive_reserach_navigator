FROM python:3.12-slim

workdir /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache \
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .


# create non-root user for running the application
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD["python", "main.py"]