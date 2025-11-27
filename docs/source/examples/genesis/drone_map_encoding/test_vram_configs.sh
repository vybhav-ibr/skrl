#!/bin/bash

# VRAM Optimization Test Script
# Tests different configurations to find optimal number of parallel environments

echo "=========================================="
echo "VRAM Optimization Testing"
echo "RTX 3050 Laptop GPU (4GB VRAM)"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python executable
PYTHON="/home/vybhav/skrl_venv/bin/python3.10"
SCRIPT_DIR="/home/vybhav/gs_gym_wrapper_reference/skrl/docs/source/examples/genesis/drone_map_encoding"

# Test function
test_config() {
    local num_envs=$1
    local script=$2
    local extra_args=$3
    local desc=$4
    
    echo -e "${YELLOW}Testing: $desc${NC}"
    echo "  Envs: $num_envs, Script: $script, Args: $extra_args"
    
    # Run for 30 seconds (about 100-200 steps)
    timeout 30s $PYTHON "$SCRIPT_DIR/$script" --num_envs=$num_envs --max_iterations=500 $extra_args 2>&1 | tail -20
    
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo -e "${GREEN}✓ SUCCESS - Ran for 30s without OOM${NC}"
        return 0
    elif [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ SUCCESS - Completed${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED - OOM or error${NC}"
        return 1
    fi
    echo ""
}

echo "=========================================="
echo "Phase 1: Testing Original Script (Fixed)"
echo "=========================================="
echo ""

test_config 1 "drone_train.py" "" "Baseline: 1 env"
test_config 2 "drone_train.py" "" "2 envs"
test_config 3 "drone_train.py" "" "3 envs"
test_config 5 "drone_train.py" "" "5 envs"

echo ""
echo "=========================================="
echo "Phase 2: Testing Optimized Script"
echo "=========================================="
echo ""

test_config 3 "drone_train_optimized.py" "" "Optimized: 3 envs (normal mode)"
test_config 5 "drone_train_optimized.py" "" "Optimized: 5 envs (normal mode)"
test_config 5 "drone_train_optimized.py" "--low_vram" "Optimized: 5 envs (low VRAM mode)"
test_config 8 "drone_train_optimized.py" "--low_vram" "Optimized: 8 envs (low VRAM mode)"

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo ""
echo "Recommendations:"
echo "  - Use the highest env count that succeeded"
echo "  - If all failed, reduce lidar resolution to 32x32"
echo "  - Monitor VRAM with: nvidia-smi -l 1"
echo ""
