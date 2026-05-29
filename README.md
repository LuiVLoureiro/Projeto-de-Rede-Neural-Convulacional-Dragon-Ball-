# Classificador Goku vs. Vegeta — CNN com ResNet50

Projeto de visão computacional que treina uma CNN (ResNet50 com transfer learning) para classificar imagens entre **Goku** e **Vegeta** do Dragon Ball, com predição em tempo real via webcam.

---

## Estrutura do Projeto

```
CNN Project/
├── data/
│   ├── raw/                  # Imagens originais (versionadas no git)
│   │   ├── goku/
│   │   └── vegeta/
│   └── processed/            # Criada automaticamente pelo preprocess.py (não versionada)
│       ├── goku/
│       └── vegeta/
├── models/                   # Criada automaticamente pelo train.py (não versionada)
│   └── goku_e_vegeta.pth
├── src/
│   ├── preprocess.py         # Redimensiona imagens brutas → processadas
│   ├── train.py              # Treina a ResNet50 e salva o modelo
│   └── predict.py            # Classifica em tempo real via webcam
└── README.md
```

> `data/processed/` e `models/` estão no `.gitignore` e **não existem após um clone**. Os scripts as criam automaticamente quando executados.

---

## Requisitos

- Python 3.9+
- PyTorch
- torchvision
- OpenCV (`cv2`)
- Pillow
- scikit-learn
- matplotlib
- seaborn

Instale todas as dependências com:

```bash
pip install torch torchvision opencv-python pillow scikit-learn matplotlib seaborn
```

> Se tiver GPU NVIDIA, instale o PyTorch com suporte a CUDA em [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Como Usar

### 0. Clone e instale as dependências

```bash
git clone https://github.com/LuiVLoureiro/Projeto-de-Rede-Neural-Convulacional-Dragon-Ball-.git
cd Projeto-de-Rede-Neural-Convulacional-Dragon-Ball-
pip install torch torchvision opencv-python pillow scikit-learn matplotlib seaborn
```

> Todos os comandos abaixo devem ser executados a partir da raiz do projeto.

---

### 1. Pré-processamento

```bash
python src/preprocess.py
```

Lê as imagens de `data/raw/`, redimensiona para **224×224** e salva em `data/processed/`. **A pasta `data/processed/` é criada automaticamente** — não é necessário criá-la manualmente.

---

### 2. Treinamento

```bash
python src/train.py
```

O script realiza duas fases:

| Fase | Descrição | Épocas |
|---|---|---|
| Feature Extraction | Apenas a camada final (fc) é treinada | 10 |
| Fine-Tuning | Camadas `layer4` + `fc` são descongeladas | 5 |

Ao final, são exibidos:
- Gráficos de loss e acurácia por época
- Matriz de confusão
- Relatório de classificação (precision, recall, F1)

O modelo é salvo automaticamente em `models/goku_e_vegeta.pth`. **A pasta `models/` é criada automaticamente** caso não exista.

**Hiperparâmetros principais** (editáveis no topo de `src/train.py`):

```python
BATCH_SIZE = 32
IMG_SIZE   = 224
EPOCHS     = 10   # Feature extraction
LR         = 1e-3
```

---

### 3. Predição em Tempo Real (Webcam)

Certifique-se de que o modelo treinado existe em `models/goku_e_vegeta.pth`, depois execute:

```bash
python src/predict.py
```

A webcam será aberta e cada frame será classificado em tempo real.

**Critério de classificação:**

| Probabilidade (sigmoid) | Resultado |
|---|---|
| ≥ 0.80 | Vegeta |
| ≤ 0.10 | Goku |
| entre 0.10 e 0.80 | Classe Desconhecida |

Pressione **`q`** para encerrar.

---

## Arquitetura do Modelo

- **Base:** ResNet50 pré-treinada no ImageNet
- **Saída:** 1 neurônio com `BCEWithLogitsLoss` (classificação binária)
- **Otimizador:** Adam
- **Divisão dos dados:** 80% treino / 20% validação (split aleatório)

**Augmentação de dados no treino:**
- Random Resized Crop
- Flip horizontal aleatório
- Rotação aleatória (±20°)
- Color Jitter (brilho e contraste)
- Normalização ImageNet

---

## Fluxo Completo

Após o clone, basta executar os três scripts em ordem:

```bash
python src/preprocess.py   # cria data/processed/ automaticamente
python src/train.py        # cria models/ e salva o .pth automaticamente
python src/predict.py      # carrega o modelo e abre a webcam
```
