base_model="Llama-3-8B-Instruct-Gradient-1048k"
attn_path=./attn_patterns/Llama-3-8B-Instruct-Gradient-1048k
model_path=./models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor


if [ ! -d "$model_path" ]; then
    mkdir -p models
    cd models
    git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor
    cd ..
fi

cd eval/needle
if [ ! -d "$model_path" ]; then
    ln -s ../../models .
fi
if [ ! -d "$attn_path" ]; then
    ln -s ../../attn_patterns .
fi
cd ../..


model_provider=LServe

# NIAH params
s_len=1000
e_len=256000
n_col=12
    
# LServe params
# For pure dense baseline, please set static_sparsity=0.0, sparse_prefill_mode=0, and sparse_decode_mode=0
static_sparsity=0.5

sparse_prefill_mode=1
precision="w8a8kv8"
kv_quant_granularity=per_tensor

sparse_decode_mode=1
dynamic_attn_budget=4096
dynamic_select_interval=4
sub_chunk_per_block=4


device=0

CUDA_VISIBLE_DEVICES=$device \
NUM_RETRIEVAL_GPU_PAGE_BLOCKS=6000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=50 \
bash eval/scripts/needle/niah_test.sh \
    $model_path $s_len $e_len $n_col \
    $model_provider $attn_path \
    $static_sparsity $sparse_prefill_mode \
    $precision $kv_quant_granularity \
    $sparse_decode_mode $dynamic_attn_budget $dynamic_select_interval $sub_chunk_per_block