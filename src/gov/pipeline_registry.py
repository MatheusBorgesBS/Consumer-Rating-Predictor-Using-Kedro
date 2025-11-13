from gov.pipelines.preprocessamento.pipeline import create_pipeline as preprocess_pipeline
from gov.pipelines.modelagem.pipeline import create_pipeline as modelagem_pipeline
from gov.pipelines.producao.pipeline import create_pipeline as producao_pipeline

def register_pipelines() -> dict:
    preprocess = preprocess_pipeline()
    modelagem = modelagem_pipeline()
    producao = producao_pipeline()

    return {
        "__default__": preprocess + modelagem + producao,
        "preprocessamento": preprocess,
        "modelagem": modelagem,
        "producao": producao
    }