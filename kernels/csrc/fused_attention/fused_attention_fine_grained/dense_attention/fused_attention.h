// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
// @article{yang2025lserve,
//   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
//   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
//   year={2025}
// }
#pragma once
#include <torch/extension.h>


torch::Tensor single_query_attention(const torch::Tensor q,
                                     const torch::Tensor k,
                                     const torch::Tensor v,
                                     c10::optional<torch::Tensor> retrieval_kv_pointers, // B x 2 x M
                                     c10::optional<torch::Tensor> streaming_kv_pointers, // B x 2 x M
                                     torch::Tensor retrieval_head_flags, // H
                                     torch::Tensor head_rank_table, // H
                                     c10::optional<const torch::Tensor> length_per_sample_,
                                     c10::optional<const torch::Tensor> alibi_slopes_,
                                     int memory_max_seqlen,
                                     int tokens_per_block,
                                     const int size_per_retrieval_token,   // default = hidden_size * sizeof(dtype)
                                     const int size_per_streaming_token,   // default = hidden_size * sizeof(dtype)
                                     const int sink_token_num, const int local_token_num,
                                     const int sink_block_num, const int local_block_num,
                                     const int num_retrieval_kv_heads,
                                     const int num_streaming_kv_heads,
                                     const int timestep,
                                     const int rotary_embedding_dim,
                                     const float rotary_base,
                                     const float rotary_embedding_scale,
                                     // neox_rotary_style = not interleaved
                                     const bool neox_rotary_style,
                                     const bool int4_kv_cache,
                                     const bool kv_cache_with_zeros,
                                     const int multiblock_switch);