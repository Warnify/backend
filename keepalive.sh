#!/bin/bash

while true;do
  curl https://backend-etxi.onrender.com/api/test
  sleep 600  # Wait for 10 minutes before sending the next request
done
