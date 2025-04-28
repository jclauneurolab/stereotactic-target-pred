#!/bin/bash


# it is running in gunicorn
TIMEOUT=60

echo "Starting gunicorn..."

cd /app/stereotactic_target_pred/stereotactic_target_pred/backend
gunicorn -b 0.0.0.0:5000 connector:app \
        --log-level=debug \
        --log-file=/dev/stdout \
        --timeout $TIMEOUT \