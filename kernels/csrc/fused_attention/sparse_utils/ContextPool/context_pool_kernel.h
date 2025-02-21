#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAGuard.h>

#include "context_pool_utils.h"

void context_paged_min_max_pool(
    at::Tensor input,              //total_seqlen x num_heads x head_size
    c10::optional<torch::Tensor> _retrieval_kv_pointers, // B x 2 x M
    at::Tensor cu_seqlens,         // b + 1
    at::Tensor pooling_heads_idx,  // num_heads
    const int max_seqlen,
    const int pooling_size,
    const int page_size,
    const int size_per_retrieval_token,   // default = hidden_size * sizeof(dtype)
    const bool kv_cache_with_zeros
);