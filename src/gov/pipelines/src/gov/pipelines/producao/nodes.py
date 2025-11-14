import pandas as pd

def limpar_dados_producao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa os dados de um novo cliente para produção.
    NÃO espera a coluna 'nota'.
    """
    df = df[
        (df['comentario'] != '<não há comentários do consumidor>')
    ].copy()
    df.loc[:, 'status'] = df['status'].map({'Resolvido': 1, 'Não Resolvido': 0})
    df = df.dropna(subset=['status', 'comentario'])
    
    return df


def transformar_novo_cliente(cliente_limpo: pd.DataFrame, vetorizador):
    X_text = vetorizador.transform(cliente_limpo['comentario'])
    X_text_df = pd.DataFrame(
        X_text.toarray(),
        columns=vetorizador.get_feature_names_out(),
        index=cliente_limpo.index
    )
    X_final = pd.concat([X_text_df, cliente_limpo['status']], axis=1)
    return X_final

def prever_nota(modelo, X_cliente: pd.DataFrame):
    predicoes = modelo.predict(X_cliente)
    probabilidades = modelo.predict_proba(X_cliente)
    resultados = []
    for i in range(len(predicoes)):
        resultado_cliente = {
            "nota_prevista": int(predicoes[i]),
            "confianca": float(max(probabilidades[i]))
        }
        resultados.append(resultado_cliente)
    print(f"Resultado da Previsão: {resultados}")
    return resultados