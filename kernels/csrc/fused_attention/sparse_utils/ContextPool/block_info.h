#pragma once


struct BlockInfo {
    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
    : sum_s(params.cu_seqlens[bidb]), actual_seqlen(params.cu_seqlens[bidb + 1] - sum_s){}

    template <typename index_t>
    __forceinline__ __device__ index_t input_offset(const index_t row_stride) const {
        return uint32_t(sum_s) * row_stride;
    }
    const int sum_s;
    const int actual_seqlen;
};