#!/bin/bash

MODEL=$1
PRECISION=$2

if [[ -z "$MODEL" || -z "$PRECISION" ]]; then
  echo "Usage: $0 <model> <precision>"
  echo "Model options: state-spaces/mamba-2.8b, state-spaces/mamba2-2.7b"
  echo "Precision options: fp16, w8a8, w4a8, w4a16"
  exit 1
fi

CMD="python main.py $MODEL --batch_size 16 --eval_zero_shot  --task_list lambada_openai --pretrained_dir ./pretrained_models"

# Append group_heads flag if model is mamba2
if [[ "$MODEL" == *"mamba2"* ]]; then
  CMD+=" --group_heads"
fi

if [[ "$MODEL" != *"quamba"* ]]; then
    case $PRECISION in
    fp16)
        ;;
    w8a8)
        # default --w_bits 8 --a_bits 8
        CMD+=" --quantize --quantize_embedding --quantize_lm_head"
        ;;
    w4a8)
        CMD+=" --quantize --w_bits 4 --a_bits 8 --apply_gptq --quantize_embedding --quantize_lm_head"
        ;;
    w4a16)
        CMD+=" --quantize --w_bits 4 --a_bits 16 --apply_gptq --quantize_embedding --quantize_lm_head"
        ;;
    *)
        echo "Unsupported precision: $PRECISION"
        exit 1
        ;;
    esac
fi

echo "Running: $CMD"
eval $CMD