#!/bin/bash

# Start the FastAPI server
gunicorn api.index:app &

# Get the process ID of the server
SERVER_PID=$!

# Polling function
while true; do
  curl -s https://warnify-backend.onrender.com/api/test
  sleep 300  # Wait for 5 minutes before sending the next request

  # Check if the server is still running
  if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server has stopped. Exiting."
    exit 1
  fi