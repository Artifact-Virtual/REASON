#!/bin/bash
# Test Artifact Reason API with a sample prime sequence problem
curl -X POST http://127.0.0.1:8000/reason \
  -H "Content-Type: application/json" \
  -d '{
    "data": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    "analysis_depth": "deep",
    "enable_multi_agent": true,
    "timeout_seconds": 60
  }' \
  > proofs/prime_analysis.json

echo "Response saved to proofs/prime_analysis.json"
