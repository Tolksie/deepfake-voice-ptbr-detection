# Deepfake Voice Detection in Brazilian Portuguese  
### MVP – Detecção de Deepfakes de Voz utilizando CNNs, RNNs, Mel-Spectrogramas e STFT

Este repositório contém todo o código-fonte, scripts, modelos e estrutura de dados utilizada no desenvolvimento do MVP do Trabalho de Conclusão de Curso intitulado:

**Análise Comparativa entre Redes Neurais Convolucionais (CNNs) e Redes Neurais Recorrentes (RNNs) na Detecção de Deepfakes de Voz em Português Brasileiro**

O objetivo do projeto é investigar a eficácia de arquiteturas clássicas de Deep Learning na distinção entre fala genuína e fala sintética em português brasileiro, utilizando representações acústicas amplamente adotadas na literatura, como Mel-Spectrogramas e STFTs.

## Conteúdo do Repositório

### `pure/`
Contém os áudios naturais utilizados no MVP. Esses arquivos derivam do corpus CETUC 16k (FalaBrasil/UFPA).  
Os responsáveis pelo corpus gentilmente concederam autorização para uso acadêmico, portanto os arquivos não foram removidos do repositório.

### `fake/`
Contém os áudios sintéticos gerados via TTS, utilizados para compor o conjunto positivo (deepfake) da classificação.

### `data/`
Inclui os dados pré-processados:  
- `dataset_prepared/` contendo os áudios padronizados  
- `features/` contendo as representações Mel e STFT  
- `labels.csv` relacionando cada arquivo ao seu respectivo rótulo (0 = natural, 1 = sintético)

### `*.py`
Scripts do MVP:
- `prepare_dataset.py` – organiza os arquivos naturais e sintéticos  
- `preprocess_audio.py` – resample, normalização e remoção de silêncio  
- `extract_features_mels.py` – gera Mel-spectrogramas padronizados  
- `extract_features_stft.py` – gera STFT padronizadas  
- `train.py` – treinamento CNN ou RNN com Mel ou STFT  
- `evaluation.py` – geração de métricas, matrizes de confusão e curvas ROC  
- `fake.py` – script utilizado originalmente para sintetizar deepfakes usando TTS

### `results/`
Contém todos os gráficos e métricas gerados pelo MVP:
- Curvas ROC  
- Matrizes de confusão  
- Gráficos de barras de métricas  
- Arquivos `.txt` contendo Accuracy, Precision, Recall, F1 e AUC

## Ambiente e Dependências

O projeto utiliza:
- Python 3.10+
- PyTorch
- Librosa
- Scikit-learn
- Matplotlib
- Numpy

Instalação:
```bash
pip install -r requirements.txt
```

## Estrutura Metodológica

O MVP segue rigorosamente o Capítulo 3 do TCC:

Dados naturais pré-processados

Dados sintéticos gerados a partir das mesmas 15 frases

Estratégia speaker-independent

Extração padronizada de Mel e STFT

Treinamento CNN vs RNN

Avaliação com Accuracy, Precision, Recall, F1 e AUC

## Licenciamento e Autorização

O corpus natural utilizado neste repositório pertence ao Grupo FalaBrasil/UFPA.
O uso aqui presente está autorizado para fins acadêmicos conforme comunicação realizada com o grupo.
Todos os arquivos originais permanecem intactos em respeito à licença BSD 2-Clause e à autorização concedida.

Autor:
Gabriel Madalena dos Santos
Universidade da Região de Joinville — UNIVILLE
Curso de Engenharia de Software
