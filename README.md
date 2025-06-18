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

## Configuração Inicial
### 1. Criação da Pasta "data"
- Crie uma pasta chamada `data` na raiz do projeto.
- Baixe o dataset IDRiD do IEEE DataPort: [IDRiD Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid).
- Extraia apenas as pastas `Segmentation` e `Disease Grading` do arquivo baixado e coloque-as dentro da pasta `data`. A estrutura deve ficar assim:
  ```
  data/
  ├── Segmentation/
  │   ├── Original Images/
  │   └── Segmentation Groundtruths/
  └── Disease Grading/
  ```

### 2. Uso de Ambiente Virtual (Recomendado)
- Para evitar conflitos entre dependências e manter o ambiente de desenvolvimento limpo, recomenda-se usar um ambiente virtual.
- **Como criar um ambiente virtual:**
  1. Abra o terminal na raiz do projeto.
  2. Execute: `python -m venv venv` (no Windows) ou `python3 -m venv venv` (no macOS/Linux).
  3. Ative o ambiente:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`
  4. Após ativar, instale as dependências conforme o arquivo 
  ```
  pip install -r requirements.txt
  ```
## Etapas do Projeto Até o Momento

### 1. Pré-processamento
- **Objetivo**: Preparar as imagens e máscaras para o treinamento do modelo U-Net.

### 2. Modelo
- **Objetivo**: Criar a arquitetura/modelo U-NET para ser usada no treinamento

### 3. Treinamento
- **Objetivo**: Treinar o modelo com as imagens "Training Set" do IDRID

## Estrutura do Código
- **Arquivo principal**: `preprocessing.py` contém o script de pré-processamento.
- **Pastas**:
  - `data/Segmentation/Original Images/Training Set`: Imagens de retina em JPG.
  - `data/Segmentation/Segmentation Groundtruths/Training Set`: Máscaras em TIF, organizadas por tipo de lesão.
- **Saída**: Arrays NumPy com imagens e máscaras pré-processadas, prontos para o treinamento do U-Net.

## Como Executar
1. Crie a pasta `data` e coloque as pastas `Segmentation` e `Disease Grading` dentro dela, conforme descrito acima.
2. Instale as dependências: `pip install -r requirements.txt` (recomenda-se usar um ambiente virtual).
3. Execute o script de pré-processamento: `python preprocessing.py`.
4. Verifique a saída: formato das imagens (54, 256, 256, 3) e máscaras (54, 256, 256, 6).

## Passo a Passo para Treinar o Modelo

### 1. Execute o seguinte comando
  ```
  python model_and_training/training.py
  ```
  Pronto, você vera o seu modelo unet_model.keras disponível no seu diretório.