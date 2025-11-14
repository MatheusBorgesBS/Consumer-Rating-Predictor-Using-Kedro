from kedro.pipeline import Pipeline, node
from .nodes import limpar_dados_producao, transformar_novo_cliente, prever_nota

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=limpar_dados_producao,  
            inputs="novo_cliente_raw",
            outputs="novo_cliente_limpo",
            name="limpar_dados_producao_node", 
        ),
        node(
            func=transformar_novo_cliente,
            inputs=["novo_cliente_limpo", "vetorizador_tfidf"], 
            outputs="X_cliente_final",
            name="transformar_novos_dados_node",
        ),
        node(
            func=prever_nota,
            inputs=["melhor_modelo", "X_cliente_final"],
            outputs="previsao_final",
            name="prever_nota_node",
        )
    ])