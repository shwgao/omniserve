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
void apply_bias_rope_update_kv_cache(const torch::Tensor qkv,
                                     torch::Tensor kv_scale_orig_quant,
                                              torch::Tensor retrieval_seq_lens,
                                              c10::optional<torch::Tensor> streaming_seq_lens,
                                              torch::Tensor padding_offset,
                                              c10::optional<torch::Tensor> retrieval_kv_pointers, // B x 2 x M
                                              c10::optional<torch::Tensor> streaming_kv_pointers, // B x 2 x M
                                              torch::Tensor retrieval_head_flags, // H
                                              torch::Tensor head_rank_table, // H
                                              // virtual sequence length (after padding)
                                              const int head_num,
                                              const int kv_head_num,
                                              const int seq_len,          // max seq len
                                              const int tokens_per_block, // default=64
                                              const int size_per_retrieval_token,   // default = hidden_size * sizeof(dtype)
                                              const int size_per_streaming_token,   // default = hidden_size * sizeof(dtype)
                                              const int sink_token_num, const int local_token_num,
                                              const int sink_block_num, const int local_block_num,
                                              const int num_retrieval_kv_heads,
                                              const int num_streaming_kv_heads,
                                              const int rotary_embedding_dim,
                                              const float rotary_embedding_base,
                                              const float rotary_embedding_scale,
                                              const int rotary_embedding_max_positions,
                                              // neox_rotary_style = not interleaved
                                              const bool neox_rotary_style,
                                              const bool int4_kv_cache,
                                              const bool kv_cache_with_zeros
                                              );
