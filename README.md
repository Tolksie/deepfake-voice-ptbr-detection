# Deepfake Voice Detection in Brazilian Portuguese (PT-BR)

### CNN × RNN usando MEL e STFT

Este repositório contém o código-fonte completo do MVP desenvolvido para o Trabalho de Conclusão de Curso “Análise comparativa entre CNNs e RNNs na detecção de deepfake de voz em Português Brasileiro”.

O objetivo é comparar arquiteturas e técnicas de extração de características, avaliando qual combinação apresenta melhor desempenho para detecção binária: voz real vs deepfake.

### 📂 Estrutura do Repositório
MVP/
│
├── data/                  # dataset preparado + features extraídas
├── models/                # pesos treinados (.pt)
├── pure/                  # áudios originais dos corpora autorizados
├── results/               # gráficos, métricas e matrizes geradas
│
├── cnn.py                 # arquitetura CNN
├── rnn.py                 # arquitetura RNN
├── dataset.py             # classe Dataset
├── train.py               # script de treinamento
├── evaluation.py          # script de avaliação e gráficos
├── preprocess_audio.py    # normalização e tratamento inicial
├── prepare_dataset.py     # separação treino/val/teste
├── extract_features_mels.py
├── extract_features_stft.py
├── fake.py                # script usado para gerar deepfakes

### 🧠 Modelos utilizados
1. CNN (Convolutional Neural Network)

Treinada nas duas representações:

✔ MEL Spectrogram

✔ STFT (Short-Time Fourier Transform)

2. RNN (LSTM – Long Short-Term Memory)

Também testada em:

✔ MEL

✔ STFT

Cada modelo possui pesos salvos em models/.

🎧 Sobre os corpora utilizados

Os áudios originais utilizados na pasta pure/ pertencem aos corpora disponibilizados pelo Grupo FalaBrasil (UFPA).

### 📌 Importante:

Os autores do corpus gentilmente concederam permissão acadêmica para uso dos dados neste projeto.
Por isso, manteremos os arquivos necessários dentro do repositório, exclusivamente para fins científicos e acadêmicos — como autorizado.

Nenhum dado sensível foi incluído.
Os arquivos permanecem organizados e sem modificações indevidas.

### 📊 Resultados (Resumo)

Foram comparados quatro pipelines:

Modelo	Feature	Acc	Prec	Rec	F1	AUC
CNN	MEL	1.000	1.000	1.000	1.000	1.00
CNN	STFT	0.97+	~1.00	~0.95	~0.97	0.997
RNN	MEL	0.70	0.66	0.82	0.73	0.915
RNN	STFT	0.99+	~1.00	~0.98	~0.99	0.996

📌 Melhor combinação geral:

⭐ CNN + MEL (desempenho perfeito no dataset)

📌 Melhor RNN:

⭐ RNN + STFT (AUC ≈ 0.996)

Todos os gráficos (AUC, confusão, métricas) estão em results/.

### ▶️ Como executar
1. Preparar dataset
python prepare_dataset.py

2. Extrair features
python extract_features_mels.py
python extract_features_stft.py

3. Treinar
python train.py

4. Avaliar
python evaluation.py

### 📝 Licença / Uso dos Dados

Este repositório é de uso exclusivamente acadêmico.
Os áudios pertencem aos autores originais, sendo usados sob permissão explícita.

### 📚 Citação

Se utilizar este código ou resultados, cite:

Santos, Gabriel M. dos.  
Deepfake Voice Detection in Brazilian Portuguese – CNN vs RNN (2025).  
GitHub: https://github.com/SEU-USUARIO/deepfake-voice-ptbr-detection
