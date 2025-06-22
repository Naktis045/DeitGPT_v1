# Use a slim Python 3.12 image based on Alpine Linux
FROM python:3.12-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# Using --no-cache-dir to save space and --upgrade pip for good measure
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application files into the container
# Assuming main.py and interphase.py are in the root of your project
COPY main.py .
COPY interphase.py .
# If you have a 'pages' directory for Streamlit, copy it too
# COPY pages/ pages/
# If you have a 'DATASET' directory for main.py, copy it
# COPY DATASET/ DATASET/

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run your Streamlit application
# Streamlit apps are typically run via 'streamlit run <your_app_file>.py'
ENTRYPOINT ["streamlit", "run", "interphase.py"]