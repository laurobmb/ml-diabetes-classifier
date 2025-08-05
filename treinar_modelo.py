"""
Script para treinar, avaliar e salvar um modelo de classificação de diabetes.

Este script executa os seguintes passos:
1. Carrega os dados de um arquivo CSV.
2. Divide os dados em conjuntos de treino e teste.
3. Aplica escalonamento de características (StandardScaler).
4. Treina um modelo RandomForestClassifier.
5. Avalia o desempenho do modelo com acurácia, relatório de classificação e matriz de confusão.
6. Salva o modelo treinado e o scaler para uso futuro.
"""
# --- IMPORTAÇÕES ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
from typing import Tuple, Any

# Importando as ferramentas necessárias da biblioteca scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONSTANTES E CONFIGURAÇÕES ---
# Definir caminhos e parâmetros em um só lugar facilita a manutenção do código.
DATA_FILE_PATH = 'csv/diabetes_data.csv'
MODEL_SAVE_PATH = 'model/modelo_diabetes.joblib'
SCALER_SAVE_PATH = 'scaler/scaler.joblib'
TARGET_COLUMN = 'outcome'       # Nome da coluna que queremos prever.
TEST_SET_SIZE = 0.2             # Usaremos 20% dos dados para teste.
RANDOM_STATE = 42               # Semente para garantir que a divisão dos dados seja sempre a mesma.

def load_data(path: str) -> pd.DataFrame:
    """Carrega o conjunto de dados de um arquivo CSV."""
    try:
        df = pd.read_csv(path)
        print(f"Arquivo '{path}' carregado com sucesso!")
        return df
    except FileNotFoundError:
        # Se o arquivo não existir, imprime uma mensagem de erro e encerra o script.
        print(f"Erro: O arquivo '{path}' não foi encontrado.")
        print("Por favor, execute primeiro o script 'gerar_csv.py' para gerar os arquivos.")
        sys.exit(1) # 'sys.exit(1)' encerra o programa indicando que houve um erro.

def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Divide os dados em características (X) e alvo (y), e depois em conjuntos de treino e teste."""
    # 'X' (features) recebe todas as colunas, EXCETO a coluna alvo. 'axis=1' indica que estamos removendo uma coluna.
    X = df.drop(TARGET_COLUMN, axis=1)
    # 'y' (target) recebe APENAS a coluna alvo.
    y = df[TARGET_COLUMN]
    
    # 'train_test_split' divide os dados.
    # 'stratify=y' é MUITO importante: garante que a proporção de cada classe (0, 1 e 2)
    # seja a mesma tanto no conjunto de treino quanto no de teste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nDados divididos em {len(X_train)} amostras de treino e {len(X_test)} amostras de teste.")
    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Ajusta e aplica o StandardScaler nas características."""
    # StandardScaler padroniza as features para que tenham média 0 e desvio padrão 1.
    scaler = StandardScaler()
    
    # --- Passo crucial para evitar "vazamento de dados" (Data Leakage) ---
    # 1. 'fit_transform': APRENDE a escala a partir dos dados de TREINO e os transforma.
    X_train_scaled = scaler.fit_transform(X_train)
    # 2. 'transform': APENAS aplica a escala JÁ APRENDIDA nos dados de TESTE.
    # Nunca usamos 'fit' nos dados de teste, pois o modelo não deve ter conhecimento prévio deles.
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def train_model(X_train_scaled: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Treina o modelo RandomForestClassifier."""
    # 'n_estimators=100' significa que nossa "floresta" terá 100 "árvores de decisão".
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    print("\nIniciando o treinamento do modelo RandomForestClassifier...")
    
    # O método 'fit' é o coração do treinamento. O modelo aprende a relação entre X_train e y_train.
    model.fit(X_train_scaled, y_train)
    print("Modelo treinado com sucesso!")
    return model

def evaluate_model(model: RandomForestClassifier, X_test_scaled: pd.DataFrame, y_test: pd.Series) -> None:
    """Avalia o modelo e imprime as métricas de desempenho."""
    # 'predict' usa o modelo treinado para fazer previsões no conjunto de teste (dados nunca vistos).
    y_pred = model.predict(X_test_scaled)
    
    print("\n--- Avaliação do Modelo ---")
    
    # Acurácia: % de previsões corretas.
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy * 100:.2f}%")
    
    print("\nRelatório de Classificação:")
    # Define os nomes das nossas classes para que o relatório fique legível.
    class_names = ['Não Diabético (0)', 'Pré-Diabético (1)', 'Diabético (2)']
    # 'classification_report' mostra métricas detalhadas (precisão, recall, f1-score) para cada classe.
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nMatriz de Confusão:")
    # A matriz mostra os acertos e erros. Linhas = real, Colunas = previsão.
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Gerando um gráfico para visualizar a Matriz de Confusão.
    plt.figure(figsize=(10, 8))
    # 'annot=True' mostra os números dentro dos quadrados. 'fmt='d'' formata como inteiros.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    # 'plt.show()' exibe a janela do gráfico.
    plt.show()

def save_artifacts(model: RandomForestClassifier, scaler: StandardScaler, model_path: str, scaler_path: str) -> None:
    """Salva o modelo e o scaler em arquivos."""
    # 'joblib' é eficiente para salvar objetos Python que contêm grandes arrays numpy, como os modelos do scikit-learn.
    print(f"\nSalvando o modelo em '{model_path}' e o scaler em '{scaler_path}'...")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Artefatos salvos com sucesso!")

def main():
    """Função principal que orquestra a execução do script."""
    # 1. Carregar dados
    df = load_data(DATA_FILE_PATH)
    # 2. Processar e dividir
    X_train, X_test, y_train, y_test = process_data(df)
    # 3. Escalonar
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    # 4. Treinar
    model = train_model(X_train_scaled, y_train)
    # 5. Avaliar
    evaluate_model(model, X_test_scaled, y_test)
    # 6. Salvar
    save_artifacts(model, scaler, MODEL_SAVE_PATH, SCALER_SAVE_PATH)

# Este é um padrão do Python. O código dentro deste 'if' só será executado
# quando o script for rodado diretamente (ex: 'python treinar_modelo.py').
if __name__ == "__main__":
    main()