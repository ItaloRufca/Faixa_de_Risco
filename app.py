
import streamlit as st
import pandas as pd
from pycaret.regression import setup, compare_models, pull, predict_model, save_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(
    page_title="Análise Preditiva de Acidentes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, sep=',', low_memory=False)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

@st.cache_resource
def run_pycaret_setup_and_compare(df, target_variable, session_id):
    setup(data=df, target=target_variable, session_id=session_id, verbose=False)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model, 'best_model_trained')
    return setup_df, compare_df, best_model

with st.sidebar:
    st.title("🚗 Análise Preditiva de Acidentes")
    st.header("1. Carregar Dados")
    uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])
    st.header("2. Selecione a Análise")
    app_mode = st.radio("Escolha a página:", [
        "Visão Geral e Análise Preditiva",
        "Previsão para Dados Futuros",
        "Arquitetura e Governança"])

if not uploaded_file:
    st.warning("Por favor, carregue o arquivo CSV na barra lateral.")
    st.stop()

df_original = load_data(uploaded_file)
if df_original is None:
    st.stop()

if app_mode == "Visão Geral e Análise Preditiva":
    st.title("🔎 Análise Preditiva com o Dataset Completo")
    st.dataframe(df_original.head())

    if st.button("Iniciar Análise Preditiva"):
        with st.spinner("Executando análise..."):
            setup_df, compare_df, best_model = run_pycaret_setup_and_compare(
                df_original, 'qtd_acidentes', 42)
            st.subheader("Configuração do Ambiente PyCaret")
            st.dataframe(setup_df)
            st.subheader("Resultados da Comparação")
            st.dataframe(compare_df)
            predictions = predict_model(best_model)
            resultado = predictions[['qtd_acidentes', 'prediction_label']]
            resultado.columns = ['Valor Real', 'Valor Previsto']
            st.dataframe(resultado.head(20))
            st.download_button("Baixar Previsões", resultado.to_csv(index=False).encode('utf-8'),
                               file_name='predicoes_holdout.csv')

elif app_mode == "Previsão para Dados Futuros":
    st.title("🔮 Previsão para Dados Futuros")
    df_historica = df_original[df_original['ano'].isin([2020, 2021, 2022])]
    df_futuro = df_original[df_original['ano'] >= 2023]
    st.dataframe(df_historica.head())
    st.dataframe(df_futuro.head())

    if st.button("Treinar e Prever"):
        with st.spinner("Treinando modelo histórico..."):
            _, _, best_model_hist = run_pycaret_setup_and_compare(df_historica, 'qtd_acidentes', 123)
        with st.spinner("Prevendo dados futuros..."):
            future_predictions = predict_model(best_model_hist, data=df_futuro)
            resultado_futuro = future_predictions[['qtd_acidentes', 'prediction_label']]
            resultado_futuro.columns = ['Valor Real (se conhecido)', 'Valor Previsto']
            st.dataframe(resultado_futuro)
            st.download_button("Baixar Previsões Futuras", resultado_futuro.to_csv(index=False).encode('utf-8'),
                               file_name='previsoes_futuras.csv')

elif app_mode == "Arquitetura e Governança":
    st.title("🏛️ Arquitetura e Governança de Dados")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    boxes = {
        "Ingestão\nFonte: Dados Abertos": (0.05, 0.7),
        "Tratamento\nLimpeza": (0.35, 0.7),
        "Modelagem\nFeatures": (0.65, 0.7),
        "Modelo ML\nPyCaret": (0.35, 0.4),
        "Dashboards\nBI": (0.65, 0.4),
        "Governança\nSegurança e Qualidade": (0.35, 0.1),
    }
    for label, (x, y) in boxes.items():
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.25, 0.15, boxstyle="round,pad=0.03",
                                            edgecolor="black", facecolor="#dbeafe"))
        ax.text(x + 0.125, y + 0.075, label, ha='center', va='center', fontsize=10, weight='bold')

    def draw_arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color='#2563eb', lw=2))

    draw_arrow((0.30, 0.775), (0.35, 0.775))
    draw_arrow((0.60, 0.775), (0.65, 0.775))
    draw_arrow((0.475, 0.7), (0.475, 0.55))
    draw_arrow((0.775, 0.7), (0.775, 0.55))
    draw_arrow((0.475, 0.4), (0.475, 0.25))

    st.pyplot(fig)
    st.markdown("[Governança baseada em DAMA-DMBOK 2.0 com segurança, acesso e rastreabilidade.]")
