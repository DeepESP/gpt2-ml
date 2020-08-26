#!/usr/bin/env bash
# Usage
# bash merge_data.sh input/folder output/file.txt

IN_FOLDER=${1}
OUT_FILE=${2}

python3 merge_data.py -input_fn ${IN_FOLDER} -output_fn ${OUT_FILE} > merge_log.txt
