# Instale as bibliotecas antes de rodar:
# pip install pycaret gradio pandas scikit-learn matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import setup, compare_models, predict_model, save_model, pull, load_model
import gradio as gr

# 1. Carregar base de dados
csv_path = "ml_acidentes_mensal_full.csv"
df = pd.read_csv(csv_path, sep=",", low_memory=False)

# 2. Corrigir tipos (km tem vírgula)
df["km"] = df["km"].astype(str).str.replace(",", ".").astype(float)

# 3. Separar histórico e atual
df_historica = df[df["ano"].isin([2020, 2021, 2022])].copy()
df_atual = df[df["ano"].isin([2023, 2024, 2025])].copy()

# 4. Setup do PyCaret
setup(data=df_historica, target="qtd_acidentes", session_id=42, silent=True, verbose=False)
best_model = compare_models()

# 5. Prever com dados recentes
predictions = predict_model(best_model, data=df_atual)
resultado = predictions[["qtd_acidentes", "prediction_label"]]
resultado.columns = ["Real", "Previsto"]

# 6. Salvar modelo e gráfico
save_model(best_model, "melhor_modelo_acidentes")

plt.figure(figsize=(10, 6))
plt.plot(resultado["Real"].values, label="Real", marker="o")
plt.plot(resultado["Previsto"].values, label="Previsto", marker="x")
plt.title("Comparação Real vs Previsto")
plt.xlabel("Índice da Amostra")
plt.ylabel("Qtd. Acidentes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparacao_real_previsto.png")
plt.close()

# 7. Gradio App
modelo = load_model("melhor_modelo_acidentes")

def prever_acidentes(ano, mes, br, km, clima_ruim, frequencia_feriado, proporcao_noite,
                     proporcao_dia_semana_fds, media_idade, qtd_infracoes):

    entrada = pd.DataFrame([{
        "ano": int(ano),
        "mes": int(mes),
        "br": float(br),
        "km": float(km),
        "clima_ruim": float(clima_ruim),
        "frequencia_feriado": float(frequencia_feriado),
        "proporcao_noite": float(proporcao_noite),
        "proporcao_dia_semana_fds": float(proporcao_dia_semana_fds),
        "media_idade_envolvidos": float(media_idade),
        "qtd_infracoes": int(qtd_infracoes)
    }])

    resultado = predict_model(modelo, data=entrada)
    return round(resultado["prediction_label"].iloc[0], 2)

interface = gr.Interface(
    fn=prever_acidentes,
    inputs=[
        gr.Number(label="Ano"),
        gr.Number(label="Mês"),
        gr.Number(label="BR"),
        gr.Number(label="KM"),
        gr.Slider(0, 1, step=0.01, label="Clima Ruim (0 a 1)"),
        gr.Slider(0, 1, step=0.01, label="Frequência de Feriado (0 a 1)"),
        gr.Slider(0, 1, step=0.01, label="Proporção de Acidentes à Noite"),
        gr.Slider(0, 1, step=0.01, label="Proporção Fim de Semana"),
        gr.Number(label="Média de Idade dos Envolvidos"),
        gr.Number(label="Qtd de Infrações")
    ],
    outputs=gr.Number(label="Previsão de Acidentes"),
    title="Modelo de Previsão de Acidentes Rodoviários",
    description="Preveja a quantidade de acidentes com base nos dados históricos tratados"
)

interface.launch()
