from kedro.pipeline import Pipeline, node
from .nodes import treinar_modelos, selecionar_e_reportar_modelos

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=treinar_modelos, 
            
            inputs=[
                "X_train",
                "y_train",
                "X_test",
                "y_test",
                "params:random_state"
            ],
            outputs="modelos_resultados", 
            name="treinar_varios_modelos_node",
        ),
        node(
            func=selecionar_e_reportar_modelos, 
            inputs="modelos_resultados",
            outputs=[
                "melhor_modelo",
                "melhor_modelo_info",
                "relatorio_todos_modelos"
            ], 
            name="selecionar_e_reportar_modelos_node",
        ),
    ])