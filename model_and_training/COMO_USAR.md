# 📋 Como Usar o Sistema de Segmentação

Este guia explica como usar os scripts de segmentação e análise do dataset.

## 🖼️ Segmentação de Imagem (`segment_image.py`)

### Descrição
Script que segmenta uma imagem de retina usando o modelo U-Net treinado e salva:
- **Imagem original**: A imagem de entrada
- **Máscara colorida**: Cada classe tem uma cor específica
- **Overlay**: Sobreposição da segmentação na imagem original

### Cores das Classes
- 🔴 **Vermelho**: Microaneurismas (MA)
- 🟢 **Verde**: Hemorragias (HE)
- 🔵 **Azul**: Exsudatos Duros (EX)
- 🟡 **Amarelo**: Exsudatos Moles (SE)
- 🟣 **Magenta**: Disco Óptico (OD)
- ⚫ **Preto**: Background (fundo)

### Como Usar

#### 1. Navegar para o diretório correto:
```powershell
cd C:\Projetos\RAS\LUZIA\model_and_training
```

#### 2. Ativar o ambiente virtual:
```powershell
..\venv\Scripts\Activate.ps1
```

#### 3. Executar a segmentação:

**Segmentar uma imagem do dataset de treino:**
```powershell
python segment_image.py "../data/idrid/Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg"
```

**Segmentar uma imagem do dataset de teste:**
```powershell
python segment_image.py "../data/idrid/Segmentation/1. Original Images/b. Testing Set/IDRiD_55.jpg"
```

**Especificar pasta de saída:**
```powershell
python segment_image.py "../data/idrid/Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg" --output meus_resultados
```

### Parâmetros
- `image_path`: Caminho para a imagem (obrigatório)
- `--output` ou `-o`: Pasta de saída (padrão: `outputs`)

### Saída
O script criará 3 arquivos na pasta de saída:
- `{nome}_original.png`: Imagem original
- `{nome}_mask.png`: Máscara colorida com legenda
- `{nome}_overlay.png`: Overlay com legenda

---

## 📊 Análise do Dataset (`analyze_dataset_distribution.py`)

### Descrição
Script que analisa a distribuição das classes no dataset IDRiD preprocessado, mostrando:
- Porcentagem de pixels por classe
- Em quantas imagens cada classe aparece
- Estatísticas detalhadas por classe

### Como Usar

#### 1. Navegar para o diretório correto:
```powershell
cd C:\Projetos\RAS\LUZIA\model_and_training
```

#### 2. Ativar o ambiente virtual:
```powershell
..\venv\Scripts\Activate.ps1
```

#### 3. Executar a análise:
```powershell
python analyze_dataset_distribution.py
```

### Saída
O script mostrará:
1. **Distribuição por Pixels**: Quantos % dos pixels totais cada classe ocupa
2. **Distribuição por Imagens**: Em quantas imagens cada classe aparece
3. **Análise Detalhada**: Estatísticas quando cada classe está presente

---

## 🛠️ Solução de Problemas

### Erro: "Modelo não encontrado"
- Certifique-se de que o arquivo `unet_model.keras` existe no diretório `model_and_training`
- Se não existir, execute o treinamento: `python training.py`

### Erro: "Imagem não encontrada"
- Verifique se o caminho da imagem está correto
- Certifique-se de que o dataset IDRiD foi extraído corretamente

### Erro: "Dados preprocessados não encontrados"
- Para o `analyze_dataset_distribution.py`, execute primeiro: `python ../preprocessing/preprocessing.py`

### Erro de ambiente Python
- Certifique-se de que o ambiente virtual está ativado
- Verifique se todas as dependências estão instaladas: `pip install -r ../requirements.txt`

---

## 📁 Estrutura de Arquivos Esperada

```
LUZIA/
├── data/
│   └── idrid/
│       └── Segmentation/
│           ├── 1. Original Images/
│           │   ├── a. Training Set/       # Imagens de treino
│           │   └── b. Testing Set/        # Imagens de teste
│           └── 2. All Segmentation Groundtruths/
├── model_and_training/
│   ├── segment_image.py                   # Script de segmentação
│   ├── analyze_dataset_distribution.py    # Script de análise
│   ├── unet_model.keras                   # Modelo treinado
│   └── outputs/                           # Resultados da segmentação
└── preprocessing/
    └── preprocessed_data/
        ├── images.npy                     # Imagens preprocessadas
        └── masks.npy                      # Máscaras preprocessadas
```

---

## 💡 Dicas

1. **Para testar rapidamente**: Use uma imagem do dataset de treino (IDRiD_01.jpg a IDRiD_54.jpg)
2. **Para ver todas as opções**: Use `python segment_image.py --help`
3. **Resultados ficam em**: `outputs/` por padrão, mas você pode mudar com `--output`
4. **Visualização**: Abra os arquivos PNG gerados para ver os resultados coloridos
