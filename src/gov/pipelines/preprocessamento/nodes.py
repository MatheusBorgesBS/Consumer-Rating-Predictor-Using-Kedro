# Em src/gov/pipelines/preprocessamento/nodes.py

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# Em: src/gov/pipelines/preprocessamento/nodes.py

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split # <-- Importe o split aqui

# --- NÓ 1 (Sem alteração) ---
def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove comentários vazios, converte tipos e limpa status."""
    df = df[
        (df['comentario'] != '<não há comentários do consumidor>') &
        (df['nota'].isin(['1', '2', '3', '4', '5']))
    ].copy()
    df['nota'] = df['nota'].astype(int)
    df.loc[:, 'status'] = df['status'].map({'Resolvido': 1, 'Não Resolvido': 0})
    df = df.dropna(subset=['status', 'comentario'])
    return df

def dividir_dados(df_limpo: pd.DataFrame, sample_frac: float, random_state: int):
    """
    Primeiro, seleciona uma amostra dos dados (se sample_frac < 1.0).
    Depois, divide em treino e teste.
    """
    
    # 1. Lógica de Amostragem (Movida para cá)
    if sample_frac < 1.0:
        df_amostrado = df_limpo.sample(frac=sample_frac, random_state=random_state)
    else:
        df_amostrado = df_limpo.copy()

    # 2. Divisão
    df_train, df_test = train_test_split(
        df_amostrado, 
        test_size=0.2, 
        random_state=random_state
    )
    return df_train, df_test

# --- NÓ 3 (Modificado) ---
def vetorizar_texto(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Vetoriza os dados de treino (fit_transform) e teste (transform)
    para evitar vazamento de dados.
    """
    
    # Carrega stopwords
    try:
        stopwords_pt = stopwords.words('portuguese')
    except LookupError:
        import nltk
        nltk.download('stopwords')
        stopwords_pt = stopwords.words('portuguese')
    stopwords_pt.extend(['<não há comentários do consumidor>', 'consumidor', 'comentários'])

    # Cria o vetorizador
    vetor = TfidfVectorizer(stop_words=stopwords_pt, max_features=500)

    # 1. TREINO: Aprende e transforma (fit_transform)
    X_train_text = vetor.fit_transform(df_train['comentario'])
    X_train_text_df = pd.DataFrame(
        X_train_text.toarray(),
        columns=vetor.get_feature_names_out(),
        index=df_train.index
    )
    # Junta a feature 'status'
    X_train = pd.concat([X_train_text_df, df_train['status']], axis=1)
    y_train = df_train['nota']

    # 2. TESTE: Apenas transforma (transform)
    X_test_text = vetor.transform(df_test['comentario'])
    X_test_text_df = pd.DataFrame(
        X_test_text.toarray(),
        columns=vetor.get_feature_names_out(),
        index=df_test.index
    )
    # Junta a feature 'status'
    X_test = pd.concat([X_test_text_df, df_test['status']], axis=1)
    y_test = df_test['nota']

    # Retorna tudo, pronto para o pipeline de modelagem
    return X_train, y_train, X_test, y_test, vetor