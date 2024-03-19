#!/bin/bash

while true;do
  curl https://backend-etxi.onrender.com/api/test
  sleep 840  # Wait for 14 minutes before sending the next request
done
