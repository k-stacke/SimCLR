#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2

OUTPUT_FOLDER='/proj/karst/results/simclr/'$FOLDER'/finetune_linear_classification_'$MODEL''
#OUTPUT_FOLDER='/proj/karst/results/simclr/'$DTime'_simclr'

python simclr/linear.py \
--model_path '/proj/karst/results/simclr/'$FOLDER'/128_0.5_200_24_2_model_'$MODEL'.pth' \
--batch_size 16 \
--epochs 2 \
--data_input_dir '/proj/karst/camelyon16/pcam' \
--save_dir $OUTPUT_FOLDER \
--save_after 1 \
--lr 0.01 \
--training_data_csv '/proj/karst/camelyon16/pcam/validation/pcam_patches_valid.csv' \
--test_data_csv '/proj/karst/camelyon16/pcam/test/pcam_patches_test.csv' \
--trainingset_split 0.75 \
--finetune
