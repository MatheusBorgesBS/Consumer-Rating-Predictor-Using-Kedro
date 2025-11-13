# Em: src/gov/pipelines/modelagem/nodes.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# --- NÓ 1 (MODIFICADO E SIMPLIFICADO) ---
def treinar_modelos(X_train, y_train, X_test, y_test, random_state: int):
    """
    Node do Kedro: treina múltiplos modelos nos dados JÁ DIVIDIDOS
    e retorna um dicionário de resultados.
    """
    
    # A lógica de sample_frac E train_test_split foi REMOVIDA DAQUI.
    # Esta função agora só recebe os dados prontos.

    # Modelos a testar
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

    resultados = {}

    # Treina e avalia
    for nome, modelo in modelos.items():
        # Treina no X_train
        modelo.fit(X_train, y_train)
        # Pontua no X_test
        score = accuracy_score(y_test, modelo.predict(X_test)) 
        
        resultados[nome] = {
            "modelo": modelo,
            "accuracy": score,
            "params": modelo.get_params()
        }
    
    return resultados


# --- NÓ 2 (Sem alteração) ---
def selecionar_e_reportar_modelos(modelos_resultados: dict):
    """
    Recebe o dicionário de resultados.
    Retorna:
    1. O objeto do melhor modelo
    2. Um DataFrame com as infos do melhor modelo
    3. Um DataFrame com o ranking de TODOS os modelos
    """
    
    report_data = []
    for nome, data in modelos_resultados.items():
        report_data.append({
            "modelo": nome,
            "accuracy": data["accuracy"]
        })
    
    df_relatorio_todos = pd.DataFrame(report_data).sort_values(
        by="accuracy", ascending=False
    ).reset_index(drop=True)

    melhor_nome = df_relatorio_todos.iloc[0]["modelo"] 
    melhor_modelo = modelos_resultados[melhor_nome]["modelo"]
    hiperparams = modelos_resultados[melhor_nome]["params"]
    accuracy = modelos_resultados[melhor_nome]["accuracy"]

    info_melhor = {
        "modelo": melhor_nome,
        "accuracy": accuracy,
        **hiperparams
    }
    df_info_melhor = pd.DataFrame([info_melhor])

    # Retorna as 3 saídas
    return melhor_modelo, df_info_melhor, df_relatorio_todos