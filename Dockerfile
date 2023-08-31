# Use the official Python base image
FROM 3.11.5-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Run the script to generate the token.json file and start your application
CMD ["streamlit", "run", "app.py"]

