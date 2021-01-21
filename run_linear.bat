@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i

SET OUTPUT_FOLDER=./training/%DTime%_simclr
REM SET MODEL=%1
REM SET FOLDER=%1
REM SET "OUTPUT_FOLDER=E:\OneDrive - Sectra\Research\2019\representation_shift\results\november2019\%FOLDER%"

ECHO %OUTPUT_FOLDER%
python simclr/linear.py ^
--model_path "E:/OneDrive - Sectra/Research/2020/cpc/results/simclr/20200728_1708_simclr/128_0.5_200_128_100_model_11.pth" ^
--batch_size 32 ^
--epochs 10 ^
--data_input_dir "E:/data/patch_camelyon/pcamv1" ^
--save_dir %OUTPUT_FOLDER% ^
--save_after 1 ^
--validate ^
--training_data_csv "E:/data/patch_camelyon/pcamv1/training/pcam_patches_training.csv" ^
--validation_data_csv "E:/data/patch_camelyon/pcamv1/validation/pcam_patches_valid.csv" ^
--test_data_csv "E:/data/patch_camelyon/pcamv1/test/pcam_patches_test.csv"

