#!/bin/bash
# Run at the start of every notebook session (/opt/venv is not persistent).
mkdir -p /root/tmp; TMPDIR=/root/tmp pip install -q transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 && python -c "import transformers,peft;print(transformers.__version__,peft.__version__)"
