#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

OUTPUT_FOLDER='/proj/karst/results/simclr/'$DTime'_simclr_multidata'

python simclr/main.py \
--dataset 'multidata' \
--batch_size 256 \
--epochs 200 \
--data_input_dir '/proj/karst/ciga/ciga_pathology' \
--save_dir $OUTPUT_FOLDER \
--save_after 1 \
--optimizer 'lars' \
--lr 0.028125 \
--training_data_csv '/proj/karst/camelyon16/camelyon16_patches_training_1000.csv' \
--test_data_csv '/proj/karst/camelyon16/camelyon16_patches_training_1000.csv' \
--validation_data_csv '/proj/karst/camelyon16/camelyon16_patches_training_1000.csv'
