# Análise Preditiva de Acidentes com Streamlit e PyCaret

Este projeto é um app interativo desenvolvido com [Streamlit](https://streamlit.io/) que utiliza a biblioteca [PyCaret](https://pycaret.org/) para realizar análises preditivas em dados de acidentes.

## 📊 Funcionalidades

- Upload de arquivo CSV com dados de acidentes
- Treinamento automático de modelos de regressão com PyCaret
- Comparação de modelos e exibição do melhor
- Previsão para dados futuros com base em histórico
- Visualização de arquitetura do pipeline e plano de governança

## 🧠 Tecnologias Usadas

- Python 3.8+
- Streamlit
- PyCaret (regressão)
- Pandas
- Matplotlib

## 🚀 Como Executar no Navegador (Streamlit Cloud)

1. Acesse o site [https://share.streamlit.io](https://share.streamlit.io)
2. Conecte sua conta do GitHub
3. Crie um novo app e selecione este repositório
4. No campo de entrada de arquivo principal, digite: `app.py`
5. Clique em **Deploy**

## 📁 Estrutura Esperada do CSV

O arquivo deve conter uma coluna `qtd_acidentes` (target) e colunas auxiliares como `ano`, `mes`, etc.

## 📦 Requisitos (requirements.txt)

```
streamlit
pandas
pycaret
matplotlib
```

## 📚 Sobre a Arquitetura

O app simula um pipeline de dados completo, com ingestão, tratamento, modelagem e visualização, além de um plano de governança baseado no DAMA-DMBOK.

## 🛡️ Conformidade

Os dados utilizados são públicos e anonimizados, em conformidade com a LGPD.

## ✍️ Autor

Projeto desenvolvido por [Seu Nome Aqui].
