@echo off
REM Este script executa a segmentação com caminhos pré-definidos.

REM Caminhos para os seus arquivos
set IMAGEM="data\Segmentation\Original Images\Testing Set\IDRiD_55.jpg"
set MODELO="model_and_training\unet_model_best.keras"
set SAIDA="validation\outputs"
set SCRIPT_PYTHON="validation\segment_image.py"

echo Iniciando segmentacao para a imagem: %IMAGEM%

REM Executa o comando Python
python %SCRIPT_PYTHON% %IMAGEM% --model %MODELO% --output %SAIDA%

echo Processo concluido.
pause