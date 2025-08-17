#!/bin/bash

#!/bin/bash
LOG_FILE="w2v_model.log"

# Run script and log everything
#./ensemble.sh 2>&1 | tee "$LOG_FILE"

# CUDA_VISIBLE_DEVICES=1 python ensemble.py 2>&1 | tee "$LOG_FILE"
# CUDA_VISIBLE_DEVICES=0 python w2v_pr_si_er.py 2>&1 | tee "$LOG_FILE"
CUDA_VISIBLE_DEVICES=1 python w2v_pr_si_er_multi_gpu.py 2>&1 | tee "$LOG_FILE"

