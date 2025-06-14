import gradio as gr
import pandas as pd
from pycaret.regression import setup, create_model, save_model, load_model, predict_model
import os

# --- Configuração Global ---
# Nome do ficheiro onde o modelo treinado será guardado no servidor.
MODEL_FILE = 'et_accident_model.pkl'

# Tenta carregar os dados. O nome do ficheiro deve corresponder ao que está no repositório.
try:
    df = pd.read_csv('ml_acidentes_mensal_full.csv', sep=',', low_memory=False)
except FileNotFoundError:
    print("ERRO: Ficheiro de dados 'ml_acidentes_mensal_full.csv' não encontrado.")
    # Exibe uma mensagem de erro na interface do Gradio se o ficheiro não for encontrado
    df = None

# --- Função de Treino do Modelo ---
def train_and_save_model():
    """
    Carrega os dados, treina um modelo Extra Trees Regressor usando PyCaret
    e guarda-o em disco. Este processo é executado apenas uma vez no servidor.
    """
    if df is None:
        raise FileNotFoundError("Não é possível treinar o modelo, o ficheiro de dados não foi carregado.")

    print(f"A iniciar o treino do modelo...")
    print("Este processo pode demorar vários minutos e só acontece na primeira execução no servidor.")
    
    # Configura o ambiente PyCaret.
    print("Passo 1/3: A configurar o ambiente PyCaret...")
    setup(data=df, target='qtd_acidentes', session_id=42, verbose=False, use_gpu=False)
    
    # Cria e treina o modelo Extra Trees, que foi o melhor no notebook original.
    print("Passo 2/3: A treinar o modelo Extra Trees Regressor...")
    et_model = create_model('et', verbose=False) 
    
    # Guarda o pipeline completo (modelo + transformações)
    model_path = MODEL_FILE.replace('.pkl', '')
    print(f"Passo 3/3: A guardar o modelo treinado como '{model_path}'...")
    save_model(et_model, model_path)
    
    print("-" * 50)
    print(f"Modelo treinado e guardado com sucesso!")
    print("-" * 50)
    return load_model(model_path)

# --- Verificação e Carregamento do Modelo ---
# No Hugging Face Spaces, o sistema de ficheiros pode ser efémero.
# O treino será executado se o ficheiro do modelo não for encontrado.
if not os.path.exists(MODEL_FILE):
    if df is not None:
        # Treina e carrega o modelo
        loaded_model = train_and_save_model()
    else:
        # Se não há dados, o modelo não pode ser treinado/carregado
        loaded_model = None
else:
    # Se o modelo já existe, simplesmente carrega-o
    print(f"A carregar modelo pré-treinado de '{MODEL_FILE}'...")
    loaded_model = load_model(MODEL_FILE.replace('.pkl', ''))
    print("Modelo carregado com sucesso.")

# --- Função Principal de Previsão ---
def predict_accident_risk(uf, br, mes, dia_semana, condicao_meteo, tipo_pista, tracado_via, uso_solo):
    """
    Recebe os inputs do utilizador, cria um DataFrame e usa o modelo
    pré-treinado para fazer uma previsão.
    """
    if loaded_model is None:
        return "Erro: Modelo não carregado.", "Verifique os logs do servidor."

    # Cria um dicionário com os dados de entrada
    input_data = {
        'uf': [uf], 'br': [br], 'mes': [mes], 'dia_semana': [dia_semana],
        'condicao_meteo': [condicao_meteo], 'tipo_pista': [tipo_pista],
        'tracado_via': [tracado_via], 'uso_solo': [uso_solo],
        # Adiciona valores padrão para colunas que o modelo espera mas não estão na UI
        'ano': [2024], 'km': [100], 'latitude': [-15.78], 'longitude': [-47.92],
        'proporcao_noite': [0.5], 'frequencia_feriado': [0.1]
    }
    
    input_df = pd.DataFrame(input_data)
    
    predictions = predict_model(loaded_model, data=input_df)
    predicted_value = predictions['prediction_label'].iloc[0]
    
    risk_level = "Baixo"
    if predicted_value > 2.5: risk_level = "Alto"
    elif predicted_value > 1.5: risk_level = "Médio"
        
    return f"{predicted_value:.2f}", {risk_level: 1.0}

# --- Construção da Interface Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚗 Demo de Previsão de Risco de Acidentes
        Introduza os detalhes de um cenário para prever a quantidade de acidentes com base num modelo de Machine Learning (Extra Trees Regressor). O modelo foi treinado com dados abertos da PRF.
        """
    )
    with gr.Row():
        with gr.Column():
            uf = gr.Dropdown(['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'], label="UF", info="Selecione o estado da ocorrência.")
            br = gr.Number(label="BR", info="Número da rodovia federal (ex: 101, 116).")
            mes = gr.Slider(minimum=1, maximum=12, step=1, label="Mês", info="Mês do ano.")
            dia_semana = gr.Dropdown(["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"], label="Dia da Semana")
        with gr.Column():
            condicao_meteo = gr.Radio(["Céu Claro", "Chuva", "Nublado", "Nevoeiro/Neblina", "Vento"], label="Condição Meteorológica")
            tipo_pista = gr.Radio(["Simples", "Dupla", "Múltipla"], label="Tipo de Pista")
            tracado_via = gr.Radio(["Reta", "Curva", "Cruzamento", "Viaduto", "Ponte"], label="Traçado da Via")
            uso_solo = gr.Radio(["Rural", "Urbano"], label="Uso do Solo na Região")
    
    with gr.Row():
        submit_btn = gr.Button("Submeter", variant="primary")

    with gr.Row():
        with gr.Column():
            output_prediction = gr.Textbox(label="Previsão de Quantidade de Acidentes")
        with gr.Column():
            output_risk = gr.Label(label="Nível de Risco")
            
    submit_btn.click(
        fn=predict_accident_risk,
        inputs=[uf, br, mes, dia_semana, condicao_meteo, tipo_pista, tracado_via, uso_solo],
        outputs=[output_prediction, output_risk]
    )

# Lança a aplicação
if __name__ == "__main__":
    demo.launch()

