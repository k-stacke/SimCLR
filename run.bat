@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i

SET OUTPUT_FOLDER=./training/%DTime%_simclr
REM SET MODEL=%1
REM SET FOLDER=%1
REM SET "OUTPUT_FOLDER=E:\OneDrive - Sectra\Research\2019\representation_shift\results\november2019\%FOLDER%"

ECHO %OUTPUT_FOLDER%
python simclr/main.py ^
--batch_size 32 ^
--epochs 2 ^
--data_input_dir "F:/data/camelyon17/slide_data202003" ^
--save_dir %OUTPUT_FOLDER% ^
--save_after 1 ^
--validate ^
--training_data_csv "F:/data/camelyon17/slide_data202003/camelyon17_patches_unbiased.csv" ^
--validation_data_csv "F:/data/camelyon17/slide_data202003/camelyon17_patches_unbiased.csv" ^
--test_data_csv "F:/data/camelyon17/slide_data202003/camelyon17_patches_unbiased.csv"

