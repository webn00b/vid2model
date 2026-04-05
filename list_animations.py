#!/usr/bin/env python3
import os
import json

output_dir = "output"
bvh_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".bvh")])
print(json.dumps(bvh_files))
