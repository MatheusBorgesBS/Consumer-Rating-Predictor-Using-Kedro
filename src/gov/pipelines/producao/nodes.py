import pandas as pd

# --- NOVO NÓ DE LIMPEZA ---
def limpar_dados_producao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa os dados de um novo cliente para produção.
    NÃO espera a coluna 'nota'.
    """
    # Filtra linhas válidas (só o comentário)
    df = df[
        (df['comentario'] != '<não há comentários do consumidor>')
    ].copy()

    # Converte 'status' (Não Resolvido = 0)
    df.loc[:, 'status'] = df['status'].map({'Resolvido': 1, 'Não Resolvido': 0})
    df = df.dropna(subset=['status', 'comentario'])
    
    return df

# --- NÓS ANTIGOS (Sem alteração) ---
def transformar_novo_cliente(cliente_limpo: pd.DataFrame, vetorizador):
    # ... (código igual ao anterior)
    X_text = vetorizador.transform(cliente_limpo['comentario'])
    X_text_df = pd.DataFrame(
        X_text.toarray(),
        columns=vetorizador.get_feature_names_out(),
        index=cliente_limpo.index
    )
    X_final = pd.concat([X_text_df, cliente_limpo['status']], axis=1)
    return X_final

def prever_nota(modelo, X_cliente: pd.DataFrame):
    """
    Usa o modelo treinado para prever a nota de TODOS os novos clientes.
    Retorna uma lista de dicionários.
    """
    # 1. Faz a previsão para TODOS os clientes de uma vez
    predicoes = modelo.predict(X_cliente)
    
    # 2. Pega as probabilidades de todos
    probabilidades = modelo.predict_proba(X_cliente)
    
    # 3. Monta a lista de resultados
    resultados = []
    for i in range(len(predicoes)):
        resultado_cliente = {
            "nota_prevista": int(predicoes[i]),
            "confianca": float(max(probabilidades[i]))
            # (Opcional) Adiciona o comentário original para referência
            # "comentario_original": X_cliente.iloc[i].name 
            # (isso só funciona se o índice foi preservado)
        }
        resultados.append(resultado_cliente)
    
    # O print agora mostrará uma lista
    print(f"Resultado da Previsão: {resultados}")
    
    # 4. Retorna a LISTA de resultados
    return resultados