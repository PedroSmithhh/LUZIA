# Projeto de Segmentação de Retinopatia Diabética com U-Net e IDRiD

## Descrição
Este projeto tem como objetivo desenvolver um sistema para segmentação de lesões relacionadas à retinopatia diabética em imagens de retina, utilizando o modelo U-Net e o dataset IDRiD (Indian Diabetic Retinopathy Image Dataset). A retinopatia diabética é uma complicação do diabetes que afeta os vasos sanguíneos da retina, e a detecção precoce de lesões como microaneurismas, hemorragias, exsudatos duros, exsudatos moles e disco óptico é essencial para o diagnóstico e tratamento.

Atualmente, o projeto está focado na etapa de **pré-processamento** dos dados, preparando as imagens e máscaras do dataset IDRiD para o treinamento do modelo U-Net.

## Estrutura do Dataset IDRiD
- **Imagens**: 54 imagens de retina em formato JPG, localizadas em `Segmentation/Original Images/Training Set`.
- **Máscaras**: Máscaras de segmentação em formato TIF, organizadas em subpastas para cada tipo de lesão:
  - Microaneurismas (MA)
  - Hemorragias (HE)
  - Exsudatos duros (EX)
  - Exsudatos moles (SE)
  - Disco óptico (OD)
- As máscaras indicam, em nível de pixel, a presença de cada lesão.

## Etapas do Projeto Até o Momento
### 1. Pré-processamento
- **Objetivo**: Preparar as imagens e máscaras para o treinamento do modelo U-Net.

## Ferramentas Utilizadas
- **Python**: Linguagem principal do projeto.
- **OpenCV**: Para carregar e processar imagens e máscaras.
- **NumPy**: Para manipular arrays de dados de forma eficiente.
- **TensorFlow**: Para conversão das máscaras para o formato one-hot.
- **Pathlib**: Para gerenciar caminhos de arquivos de forma robusta.
