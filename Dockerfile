# Use the official Python image.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "python check_db.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
