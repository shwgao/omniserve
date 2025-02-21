cd eval/needle
mkdir -p logs img results

model_path=$1
s_len=$2
e_len=$3
num_len=$4
model_provider=$5
attn_path=$6
static_sparsity=$7
sparse_prefill_mode=$8
precision=$9
kv_quant_granularity=${10}
sparse_decode_mode=${11}
dynamic_attn_budget=${12}
dynamic_select_interval=${13}
sub_chunk_per_block=${14}

ctx_sink_token=128
ctx_local_token=8192
dec_sink_token=128
dec_local_token=256

suffix=sparse_prefill_mode_${sparse_prefill_mode}_${precision}_${kv_quant_granularity}_static_sparsity_${static_sparsity}_sparse_decode_mode_${sparse_decode_mode}_dynamic_attn_budget_${dynamic_attn_budget}_dynamic_select_interval_${dynamic_select_interval}
if [ "$sparse_prefill_mode" == "1" ]; then
    suffix=${suffix}_context_S${ctx_sink_token}L${ctx_local_token}
elif [ "$sparse_prefill_mode" = "0" ]; then
    suffix=${suffix}
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']. Now "$sparse_prefill_mode. 
fi
echo "suffix: $suffix"

model_name=$(basename "$model_path")

common_args="--s_len $s_len \
    --e_len $e_len \
    --context_lengths_num_intervals $num_len \
    --model_provider $model_provider \
    --model_name_suffix $suffix \
    --method $method \
    --model_path $model_path \
    --ifb-mode \
    --precision $precision \
    --quant-path $model_path \
    --group-size -1 \
    --max-num-batched-tokens 4195000 \
    --max-num-seqs 1 \
    --omit-prompt \
    --kv-quant-granularity $kv_quant_granularity \
    --chunk-prefill-size 32000 \
    --multiblock-switch 32000 \
    --static-sparse-attn-load-dir $attn_path \
    --static-sparsity $static_sparsity \
    --sparse-decode-mode $sparse_decode_mode \
    --ctx-sink-token $ctx_sink_token \
    --ctx-local-token $ctx_local_token \
    --dec-sink-token $dec_sink_token \
    --dec-local-token $dec_local_token \
    --sub-chunk-per-block $sub_chunk_per_block \
    --dynamic-sparse-token-budget $dynamic_attn_budget \
    --selector-update-interval $dynamic_select_interval"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$sparse_prefill_mode" == "1" ]; then
    python -u needle_in_haystack.py $common_args --sparse-context-mode
elif [ "$sparse_prefill_mode" = "0" ]; then
    python -u needle_in_haystack.py $common_args
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']. Now "$sparse_prefill_mode. 
fi 2>&1 | tee logs/eval_${model_name}_${suffix}.log

python visualize.py \
    --folder_path "results/${model_name}_${suffix}/" \
    --model_name "${model_name} ${suffix}" \
    --pretrained_len 256000

