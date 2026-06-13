module purge 2>/dev/null || true
module load cuda/12.3
module load openmpi/4.1.6_ucx1.14.1_cuda12.3

# NOTE: 不要手動覆蓋 CUDA_HOME / PATH / LD_LIBRARY_PATH。
# cuda/12.3 module 已正確設定 CUDA_HOME=/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.3
# 並把 bin/lib64 加進 PATH/LD_LIBRARY_PATH。先前硬寫 /work/opt/t2_r8/cuda/12.3 是錯的路徑，
# 會導致 Makefile 的 -L$(CUDA_HOME)/lib64 找不到 libcudart。
echo "[env] CUDA_HOME=$CUDA_HOME"
echo "[env] nvcc:   $(command -v nvcc   || echo 'NOT FOUND')"
echo "[env] mpicxx: $(command -v mpicxx || echo 'NOT FOUND')"
