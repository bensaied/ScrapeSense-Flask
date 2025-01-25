FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y python3-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]