MODEL_PATH=./qserve_checkpoints/Llama-3-8B-Instruct-QServe # Please set the path accordingly


common_args="--max-num-batched-tokens 4195000 \
             --chunk-prefill-size 1024000 \
             --sparse-decode-mode 0"


NUM_RETRIEVAL_GPU_PAGE_BLOCKS=12000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=0 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_e2e_generation.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path $MODEL_PATH \
  --group-size -1 \
  --max-num-seqs 64 \
  --kv-quant-granularity fine_grained $common_args