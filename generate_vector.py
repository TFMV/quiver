#!/usr/bin/env python3
import json
import numpy as np

# Generate a 128-dimensional vector with values between 0 and 1
vector = np.random.rand(128).tolist()

# Create the payload
payload = {"vector": vector, "id": 12345, "metadata": {"text": "This is a test vector"}}

# Print the JSON payload
print(json.dumps(payload))
