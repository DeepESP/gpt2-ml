#!/usr/bin/env bash

NUM_FOLDS=1024
MAX_SEQ_LENGTH=10240
FN=${1}
OUT_BUCKET=${2}

rm -rf logs_${MAX_SEQ_LENGTH}
mkdir logs_${MAX_SEQ_LENGTH}
parallel -j $(nproc --all) --will-cite "python prepare_data_esp.py -fold {1} -num_folds ${NUM_FOLDS} -base_fn ${OUT_BUCKET}/data_${MAX_SEQ_LENGTH}/ -input_fn ${FN} -max_seq_length ${MAX_SEQ_LENGTH} > logs_${MAX_SEQ_LENGTH}/log{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))
