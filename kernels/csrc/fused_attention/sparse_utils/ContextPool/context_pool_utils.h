#include "../../common/kvCacheUtils.h"

struct Context_pool_params
{
    using index_t = int64_t;
    half * input_ptr;
    // int64_t * o_ptrs;
    int * cu_seqlens;
    int * pooling_heads_idx;

    index_t input_row_stride;
    index_t input_head_stride;
    // index_t o_ptrs_batch_stride;

    int b, d, max_seqlen_rounded, pooling_size, page_size;

    int input_h, pool_h;

    mutable KVBlockArray<false> kv_buffer;
};

