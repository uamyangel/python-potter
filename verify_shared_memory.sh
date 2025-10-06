#!/bin/bash
# 
# Shared Memory Verification Script for Ubuntu
# 验证共享内存是否正确工作
#

set -e

echo "============================================================"
echo "Shared Memory Verification Script"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check system memory
echo "1. System Memory Check:"
echo "---"
free -h
echo ""

AVAILABLE_GB=$(free -g | awk '/^Mem:/{print $7}')
echo "Available memory: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 30 ]; then
    echo -e "${RED}⚠️  Warning: Less than 30 GB available. Close other programs.${NC}"
else
    echo -e "${GREEN}✅ Sufficient memory available${NC}"
fi
echo ""

# 2. Check Python version
echo "2. Python Version Check:"
echo "---"
python --version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

if [ "${PYTHON_VERSION}" \< "3.8" ]; then
    echo -e "${RED}❌ Python 3.8+ required for shared_memory support${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Python version compatible${NC}"
fi
echo ""

# 3. Check required modules
echo "3. Required Modules Check:"
echo "---"
python -c "
import sys
missing = []

try:
    import numpy
    print(f'✅ numpy {numpy.__version__}')
except ImportError:
    missing.append('numpy')
    print('❌ numpy')

try:
    import numba
    print(f'✅ numba {numba.__version__}')
except ImportError:
    missing.append('numba')
    print('❌ numba')

try:
    import psutil
    print(f'✅ psutil {psutil.__version__}')
except ImportError:
    print('⚠️  psutil (optional, for memory monitoring)')

try:
    from multiprocessing import shared_memory
    print('✅ multiprocessing.shared_memory')
except ImportError:
    missing.append('shared_memory')
    print('❌ multiprocessing.shared_memory')

if missing:
    print(f'\\nMissing modules: {missing}')
    sys.exit(1)
"
echo ""

# 4. Run minimal shared memory test
echo "4. Running Minimal Shared Memory Test:"
echo "---"
python test_shared_memory_minimal.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Minimal test passed${NC}"
else
    echo -e "${RED}❌ Minimal test failed${NC}"
    exit 1
fi
echo ""

# 5. Check shared memory filesystem
echo "5. Shared Memory Filesystem:"
echo "---"
echo "Before test:"
ls -lh /dev/shm/ | head -10
SHM_BEFORE=$(du -sh /dev/shm/ | awk '{print $1}')
echo "Current usage: $SHM_BEFORE"
echo ""

# 6. Recommendations
echo "============================================================"
echo "Recommendations for Full Test:"
echo "============================================================"
echo ""
echo "Start with small configuration:"
echo -e "${YELLOW}python main.py -i benchmarks/logicnets_jscl_unrouted.phys -o test_output.phys --device_cache light -t 4 --max_iter 5${NC}"
echo ""
echo "Monitor memory in another terminal:"
echo -e "${YELLOW}watch -n 1 'ps aux | grep python | grep -v grep | awk \"{sum+=\\\$6} END {print \\\"Total RSS: \\\" sum/1024/1024 \\\" GB\\\"}'${NC}"
echo ""
echo "Expected memory usage:"
echo "  4 processes:  ~10 GB"
echo "  8 processes:  ~16 GB"
echo "  16 processes: ~28 GB"
echo "  24 processes: ~40 GB"
echo ""
echo "⚠️  If memory > 50 GB with 16 processes, shared memory is NOT working!"
echo ""
echo "============================================================"
echo "Ready to test!"
echo "============================================================"
