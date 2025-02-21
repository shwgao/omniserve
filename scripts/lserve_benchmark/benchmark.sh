# Download the model config files for benchmarking
MODEL_CONFIG_DIR_PATH=./QServe-benchmarks
if [ ! -d "$MODEL_CONFIG_DIR_PATH" ]; then
    git clone https://huggingface.co/datasets/mit-han-lab/QServe-benchmarks
fi


model_path=($1)     # Path to the model checkpoint
attn_path=($2)      # Offline calibrated attention head importance score (by DuoAttn)

# Benchmark settings
batch_size=($3)
prompt_len=($4)
decode_len=($5)

# Quantization config
precision=($6)
kv_quant_granularity=($7)           # Granularity of kv quantization. Choose from ['per_tensor', 'fine_grained']

# Sparsity config
static_sparsity=($8)                # Sparsity for static sparse attention, the ratio of attention heads to prune to streaming.
sparse_prefill_mode=($9)            # Whether to activate sparse prefilling. Choose from ['0', '1'].
sparse_decode_mode=(${10})          # Whether to activate dynamic sparse attention. Choose from ['0', '1'].
dynamic_attn_budget=${11:-4096}     # Token budget for dynamic sparse attention. 
dynamic_select_interval=${12:-4}    # The interval of steps to activate the dynamic page selector
sub_chunk_per_block=${13:-4} 

# GPU index
device_id=${14:-0}


export GLOBAL_BATCH_SIZE=${batch_size} 
export GLOBAL_PROMPT_LEN=${prompt_len} 
export GLOBAL_GENERATE_LEN=${decode_len} 
export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=5000 
export NUM_STREAMING_GPU_PAGE_BLOCKS=500
export CUDA_VISIBLE_DEVICES=${device_id} 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common_args="--model $model_path \
             --benchmarking \
             --precision ${precision} \
             --group-size -1 \
             --max-num-batched-tokens 4195000 \
             --chunk-prefill-size 32000 \
             --kv-quant-granularity $kv_quant_granularity \
             --multiblock-switch 2048 \
             --static-sparse-attn-load-dir $attn_path \
             --static-sparsity $static_sparsity \
             --sparse-decode-mode $sparse_decode_mode \
             --ctx-sink-token 128 \
             --ctx-local-token 8192 \
             --dec-sink-token 128 \
             --dec-local-token 256 \
             --sub-chunk-per-block $sub_chunk_per_block \
             --dynamic-sparse-token-budget $dynamic_attn_budget \
             --selector-update-interval $dynamic_select_interval"

if [ "$sparse_prefill_mode" = "1" ]; then
    python lserve_benchmark.py $common_args --sparse-context-mode
elif [ "$sparse_prefill_mode" = "0" ]; then
    python lserve_benchmark.py $common_args
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']."
fi
