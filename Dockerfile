FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY /requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the entire project directory into the container
COPY . .

EXPOSE 8000

# Start Uvicorn with the correct location of the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]