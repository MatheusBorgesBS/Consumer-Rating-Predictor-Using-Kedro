from kedro.pipeline import Pipeline, node
from .nodes import limpar_dados, dividir_dados, vetorizar_texto

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=limpar_dados,
            inputs="dados_brutos", 
            outputs="dados_limpos",
            name="limpar_dados_node",
        ),
        node(
            func=dividir_dados,
            inputs=["dados_limpos", "params:sample_frac", "params:random_state"], 
            outputs=["df_train", "df_test"],
            name="dividir_dados_node",
        ),
        node(
            func=vetorizar_texto,
            inputs=["df_train", "df_test"],
            outputs=[
                "X_train", "y_train",
                "X_test", "y_test",
                "vetorizador_tfidf"
            ],
            name="vetorizar_texto_node",
        )
    ])