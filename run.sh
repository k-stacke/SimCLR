#!/bin/bash
set -e

DTime=$( date +%Y%m%d_%H%M )
OUTPUT_FOLDER='./test_results/'$DTime'_simclr'

python main.py \
    --batch_size 8 \
    --epochs 2 \
    --data_input_dir '/home/ka-stk/data/slide_data202003' \
    --save_dir $OUTPUT_FOLDER \
    --save_after 1 \
    --validate \
    --training_data_csv 'test_results/20200720_1552_simclr/training_patches.csv' \
    --test_data_csv 'test_results/20200720_1552_simclr/test_patches.csv'
