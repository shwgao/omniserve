# Debug Log: OmniServe Benchmark Issues

## 1. GPU Detection Issue ✅
- **Problem:** The script did not recognize H100 GPUs, resulting in "Unsupported GPU" and incorrect device naming.
- **Solution:** Updated the detection logic to support H100. The device is now correctly reported as `H100_SXM`.
- **Status:** Fixed

## 2. UnboundLocalError in Model Runner ✅
- **Problem:** The code referenced `num_retrieval_gpu_blocks` before assignment if the environment variable was not set, causing an `UnboundLocalError`.
- **Solution:** Provided default values (`5000` for retrieval, `500` for streaming) if the environment variables are not set.
- **Status:** Fixed

## 3. CUDA Error Progression ☑️
- **Original Error:**
  - `RuntimeError: CUDA error: an illegal memory access was encountered`
  - Occurred in the dynamic sparse attention CUDA kernel.
- **After Fixes:**
  - Now replaced by: `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx(...)`
  - This occurs during a `torch.nn.Linear` operation (likely in the output projection of attention).
- **Status:** Partially resolved (original error fixed, new error emerged)
- **Partially resolved:** I blocked the sparse mode by setting `sparse_prefill_mode=0` and `sparse_decode_mode=0`.
- Debugging Suggestions for CUBLAS Error
    - **Check for NaNs/Infs:** Add debug prints to check for NaNs or Infs in `attn_output` before the failing `self.o_proj(attn_output)`.
    - **Reduce batch/sequence size:** Try running with smaller `batch_size`, `prompt_len`, or `decode_len` to rule out OOM.
    - **Check dtype:** Ensure all tensors use supported dtypes for H100 (bfloat16 is supported, but check PyTorch/CUDA versions).
    - **Update PyTorch/CUDA:** Make sure the latest versions are installed for H100 support.


## Files Modified
- `lserve_benchmark.py` - Added H100 GPU detection support
- `omniserve/worker/model_runner.py` - Added default values for GPU block configuration

## Next Steps
- Replace deprecated `torch.range` with `torch.arange`.
- Add debug code to check for NaNs/Infs before the failing linear layer.
- If the CUBLAS error persists, further investigate tensor shapes and memory usage.