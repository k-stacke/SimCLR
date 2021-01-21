#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

OUTPUT_FOLDER='/proj/results/simclr/'$DTime'_simclr'

python simclr/main.py \
--batch_size 2 \
--epochs 2 \
--data_input_dir '/proj/karst/slide_data202003' \
--save_dir $OUTPUT_FOLDER \
--save_after 1 \
--validate \
--training_data_csv '/proj/karst/results/datasets/training_patches_exl_val.csv' \
--test_data_csv '/proj/karst/results/validation_patches.csv'
