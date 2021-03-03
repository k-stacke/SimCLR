#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2
SEED=$3

OUTPUT_FOLDER='/proj/karst/results/simclr/'$FOLDER'/finetune_linear_classification_'$MODEL'_lr0001'
#OUTPUT_FOLDER='/proj/karst/results/simclr/'$DTime'_simclr'

python simclr/linear.py \
--model_path '/proj/karst/results/simclr/'$FOLDER'/128_0.5_200_256_100_model_'$MODEL'.pth' \
--batch_size 128 \
--epochs 20 \
--data_input_dir '/proj/karst/camelyon16/pcam' \
--save_dir ''$OUTPUT_FOLDER'/dataset_'$SEED'' \
--save_after 1 \
--lr 0.0001 \
--training_data_csv '/proj/karst/camelyon16/pcam/validation/pcam_patches_valid.csv' \
--test_data_csv '/proj/karst/camelyon16/pcam/test/pcam_patches_test.csv' \
--trainingset_split 0.75 \
--seed $SEED \
--finetune
