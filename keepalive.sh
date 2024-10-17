#!/bin/bash

# Start your main application (adjust the command as needed)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.index:app &

# Get the process ID of the server
SERVER_PID=$!

# Polling function
while true; do
  # Make an API call to your test endpoint
  curl -s https://backend-etxi.onrender.com/api/test
  
  # Wait for 100 seconds before the next request
  sleep 100

  # Check if the server is still running
  if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server has stopped. Exiting."
    exit 1
  fi
done