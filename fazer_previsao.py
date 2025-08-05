import joblib
import numpy as np
import pandas as pd

# Tenta carregar o modelo e o scaler que foram salvos pelo script de treinamento.
try:
    model = joblib.load('model/modelo_diabetes.joblib')
    scaler = joblib.load('scaler/scaler.joblib')
    print("Modelo e scaler carregados com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivos do modelo ou scaler não encontrados.")
    print("Por favor, execute o script 'treinar_modelo.py' primeiro.")
    exit()

# --- DADOS DO NOVO PACIENTE ---
# <<< ALTERAÇÃO ESSENCIAL: ATUALIZAÇÃO DO PACIENTE DE TESTE >>>
# Usamos as novas colunas 'pressao_diastolica' e 'pressao_sistolica'
# com os valores que você sugeriu.
novo_paciente = {
    'pregnancies': 2,
    'glucose': 180,                  # Glicose um pouco alta
    'pressao_diastolica': 80,        # Valor que você forneceu
    'pressao_sistolica': 130,        # Valor que você forneceu
    'skin_thickness': 29,
    'insulin': 0,                    # Valor comum para 'não medido'
    'bmi': 29.2,                     # IMC na faixa de obesidade
    'pedigree': 0.58,
    'age': 42
}

# O resto do script funciona sem alterações!
# 1. Prepara os dados
input_df = pd.DataFrame([novo_paciente])
input_scaled = scaler.transform(input_df)

# 2. Faz a previsão
previsao = model.predict(input_scaled)
probabilidades = model.predict_proba(input_scaled)

# 3. Exibe os resultados
print("\n--- Dados do Novo Paciente ---")
print(input_df.to_string(index=False))

print("\n--- Resultados da Previsão ---")
if previsao[0] == 0:
    resultado_final = "Não Diabético"
elif previsao[0] == 1:
    resultado_final = "Pré-Diabético"
else:
    resultado_final = "Diabético"

print(f"Previsão do Modelo: {resultado_final} (Classe {previsao[0]})")

print(f"\nConfiança da Previsão (Probabilidade):")
print(f"  - Chance de NÃO SER DIABÉTICO (0): {probabilidades[0][0] * 100:.2f}%")
print(f"  - Chance de SER PRÉ-DIABÉTICO (1):  {probabilidades[0][1] * 100:.2f}%")
print(f"  - Chance de SER DIABÉTICO (2):      {probabilidades[0][2] * 100:.2f}%")