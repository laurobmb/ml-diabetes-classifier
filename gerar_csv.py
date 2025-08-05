# --- Importação de Bibliotecas ---
# A biblioteca 'pandas' é a ferramenta padrão em Python para manipulação de dados em formato de tabela (DataFrames).
# A biblioteca 'numpy' é fundamental para computação numérica, permitindo criar e manipular arrays (listas de números) de forma muito eficiente.
import pandas as pd
import numpy as np

# --- Bloco de Configuração e Reprodutibilidade ---

# A "semente" (seed) do gerador de números aleatórios é definida aqui.
# Ao fixar a semente com um número (qualquer número, 42 é uma convenção popular), garantimos
# que toda vez que este script for executado, os mesmos números "aleatórios" serão gerados.
# Isso é VITAL para a ciência e para o desenvolvimento, pois permite que os resultados sejam reproduzíveis.
np.random.seed(42)

# Definição de uma constante para o número de amostras (pacientes).
# Usar uma constante no topo do arquivo é uma boa prática de programação. Se quisermos gerar
# 10.000 ou 500 dados no futuro, só precisamos alterar este valor em um único lugar.
NUM_SAMPLES = 250000


# --- Bloco de Geração de Características (Features) ---
# Aqui, criamos os dados para cada uma das colunas (características) do nosso dataset.
# As faixas de valores são inspiradas em dados médicos reais para dar plausibilidade ao nosso modelo.

pregnancies = np.random.randint(0, 17, NUM_SAMPLES)             # Número de gestações
glucose = np.random.randint(44, 200, NUM_SAMPLES)               # Nível de glicose no sangue
skin_thickness = np.random.randint(7, 100, NUM_SAMPLES)         # Espessura da prega cutânea em mm
insulin = np.random.randint(14, 850, NUM_SAMPLES)               # Nível de insulina no sangue
bmi = np.random.uniform(18.2, 67.1, NUM_SAMPLES).round(1)       # Índice de Massa Corporal (IMC)
pedigree = np.random.uniform(0.08, 2.42, NUM_SAMPLES).round(3)  # Função de pedigree de diabetes (histórico familiar)
age = np.random.randint(21, 81, NUM_SAMPLES)                    # Idade em anos

# Geração realista da pressão arterial.
# Primeiro, geramos a pressão diastólica (o valor menor).
pressao_diastolica = np.random.randint(50, 101, NUM_SAMPLES)
# Em seguida, garantimos que a pressão sistólica seja sempre maior que a diastólica,
# somando um valor aleatório à diastólica já gerada.
pressao_sistolica = pressao_diastolica + np.random.randint(20, 51, NUM_SAMPLES)


# --- Bloco da Lógica de Negócio (Criação do Alvo) ---
# Para que o modelo de Machine Learning tenha o que aprender, precisamos criar uma relação
# lógica entre as características e o resultado (outcome).
# Aqui, criamos um "modelo de risco" simplificado baseado em regras médicas comuns.

# Primeiro, definimos o que é "pressão alta".
# A condição é Verdadeira (1) se a sistólica for maior que 130 OU (|) a diastólica for maior que 80.
pressao_alta = ((pressao_sistolica > 130) | (pressao_diastolica > 80)).astype(int)

# Agora, calculamos uma pontuação de risco (score) para cada paciente.
# Cada fator de risco tem um "peso" diferente na pontuação final.
score = (
    (glucose > 140).astype(int) * 0.45 +  # Glicose alta tem o maior peso (45%)
    (bmi > 30).astype(int) * 0.25 +      # Obesidade tem um peso significativo (25%)
    (pressao_alta) * 0.15 +              # Pressão alta contribui com 15%
    (age > 40).astype(int) * 0.15        # Idade avançada também contribui com 15%
)

# Com base na pontuação de risco, classificamos cada paciente em uma das 3 categorias.
# A função 'np.select' é ideal para isso, funcionando como um if/elif/else em massa.
conditions = [
    (score >= 0.7),      # Se a pontuação for muito alta (>= 0.7)...
    (score >= 0.35)      # Senão, se a pontuação for média (>= 0.35)...
]
# ...atribua o valor correspondente da lista 'choices'.
choices = [2, 1]  # ...o paciente é 'Diabético' (2). ...o paciente é 'Pré-Diabético' (1).

# O 'default=0' é o 'else' final. Se nenhuma condição acima for atendida...
# ...o paciente é 'Não Diabético' (0).
outcome = np.select(conditions, choices, default=0)

# --- Bloco de Montagem e Exportação do DataFrame ---

# Criamos um dicionário Python. As chaves serão os nomes das colunas no nosso arquivo final.
# Os valores são os arrays de dados que geramos anteriormente.
data = {
    'pregnancies': pregnancies,
    'glucose': glucose,
    'pressao_diastolica': pressao_diastolica,
    'pressao_sistolica': pressao_sistolica,
    'skin_thickness': skin_thickness,
    'insulin': insulin,
    'bmi': bmi,
    'pedigree': pedigree,
    'age': age,
    'outcome': outcome  # A nossa coluna alvo, que o modelo tentará prever.
}

# Convertendo o dicionário em um DataFrame do Pandas, que é a estrutura de tabela padrão.
df = pd.DataFrame(data)

# Salvando o DataFrame em um arquivo no formato CSV (Comma-Separated Values).
# O parâmetro 'index=False' é importante para não salvar os índices da tabela (0, 1, 2...) como uma coluna extra no arquivo.
df.to_csv('csv/diabetes_data.csv', index=False)


# --- Verificação Final ---
# Imprimimos uma mensagem de sucesso para o usuário.
print(f"O arquivo 'csv/diabetes_data.csv' com {NUM_SAMPLES} entradas foi criado com sucesso!")

# É uma boa prática verificar a distribuição das classes geradas.
# Isso nos ajuda a saber se o dataset está muito desbalanceado (ex: muito mais de uma classe que de outra).
print("\nDistribuição das classes (0=Não, 1=Pré, 2=Diabético):")
print(df['outcome'].value_counts().sort_index())