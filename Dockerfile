# syntax=docker/dockerfile:1.4

# Choose a python version that you know works with your application
FROM python:3.12.1

# WORKDIR 

COPY --link requirements.txt .
# Install the requirements
RUN pip install -r requirements.txt

# You may copy more files like csv, images, data
COPY --link marimonotebook .
# COPY . .

EXPOSE 8080

# Create a non-root user and switch to it
RUN useradd -m app_user
USER app_user

CMD [ "marimo", "run", "ValueOfFlexibility.py", "--host", "0.0.0.0", "-p", "8080" ]

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1