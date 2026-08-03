# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /code

# 3. Install system libraries for image processing (OpenCV/PIL)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install your libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the whole project
COPY . .

# 6. Open port 3000 (Matches your app)
EXPOSE 3000

# 7. Start the app from the /api folder
CMD ["python", "api/app.py"]
