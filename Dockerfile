# Use official lightweight Python image
FROM python:3.10-slim

# Set work directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Hugging Face will map it automatically)
EXPOSE 7860

# Run your Flask app
CMD ["python", "app.py"]
