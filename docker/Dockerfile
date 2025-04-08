# Stage 1: Frontend builder
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend

# Copy only package files first for caching install step
COPY stereotactic_target_pred/frontend/package*.json ./

RUN npm install

# Now copy the rest of the frontend code
COPY stereotactic_target_pred/frontend/ .

# Build the frontend
RUN npm run build

# Stage 2: Final backend image
FROM python:3.10-slim

# Set environment variables for Poetry
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_CACHE_DIR=/opt/.cache

USER root

RUN mkdir /opt/image-build && chmod 777 /opt/image-build

COPY docker/apt-install.sh /opt/image-build
WORKDIR /opt/image-build
RUN ./apt-install.sh bash sssd acl gunicorn gcc g++ libcurl4-openssl-dev libssl-dev git

# Install Poetry in a virtual environment
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==$POETRY_VERSION

# Add Poetry to the PATH
ENV PATH="$POETRY_VENV/bin:$PATH"

# Verify Poetry installation
RUN poetry --version
RUN which poetry

# Copy application files
COPY config /app/stereotactic_target_pred/config
COPY uploads /app/stereotactic_target_pred/uploads
COPY resources /app/stereotactic_target_pred/resources
COPY stereotactic_target_pred/backend /app/stereotactic_target_pred/stereotactic_target_pred/backend
COPY poetry.lock /app/stereotactic_target_pred
COPY pyproject.toml /app/stereotactic_target_pred

WORKDIR /app/stereotactic_target_pred

# Install dependencies using Poetry
RUN poetry install --no-root

ARG CACHEBUST_FRONTEND=1
COPY --from=frontend-builder /app/frontend/dist /app/stereotactic_target_pred/stereotactic_target_pred/backend/client

EXPOSE 5000

# Setup startup script
ADD docker/startup.sh /server/
RUN chmod a+x /server/startup.sh

# Start the server
ENTRYPOINT [ "/bin/bash", "-c", "exec /server/startup.sh \"${@}\"", "--" ]
