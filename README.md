[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

# Consumer Rating Predictor com Kedro

Este projeto é um pipeline de Machine Learning de ponta a ponta construído com [Kedro](https://kedro.org). O objetivo é prever a nota (de 1 a 5) que um consumidor dará em uma reclamação no site [Consumidor.gov.br](https://Consumidor.gov.br).

O modelo usa como *features* o texto do **comentário** final do cliente e o **status** do caso (Resolvido/Não Resolvido) para prever a **nota**.

Este projeto demonstra um fluxo completo de MLOps, incluindo:
* Limpeza e pré-processamento de dados.
* Vetorização de texto (TF-IDF) sem vazamento de dados (Data Leakage).
* Treinamento e comparação de múltiplos modelos (LogisticRegression, RandomForest, GradientBoosting).
* Criação de um pipeline de **`producao`** (inferência) que carrega os artefatos salvos (`.pkl`) para prever a nota de novos clientes.

## 🚀 Visualização do Pipeline (Kedro Viz)

Abaixo está a visualização do fluxo de trabalho completo do projeto, mostrando como os dados fluem desde a origem até os relatórios finais.

**[COLE A IMAGEM DO SEU KEDRO VIZ AQUI]**

*(Para gerar essa imagem, rode `kedro viz` no seu terminal e tire um print!)*

## 🧱 Estrutura do Projeto

O projeto é dividido em três pipelines principais, registrados no `pipeline_registry.py`:

* **`preprocessamento`**: Carrega os dados brutos (`.json`), limpa o texto, aplica a amostragem (`params:sample_frac`), divide em treino/teste e vetoriza o texto, salvando o `vetorizador_tfidf` e os dados de treino/teste.
* **`modelagem`**: Consome os dados de treino/teste, treina múltiplos modelos, gera um relatório (`relatorio_todos_modelos.csv`) comparando a acurácia de todos e salva o melhor modelo (`melhor_modelo.pkl`).
* **`producao`**: Um pipeline de inferência independente. Ele carrega um novo arquivo (`cliente_para_prever.json`), usa o `vetorizador_tfidf` e o `melhor_modelo` salvos para fazer a previsão e salva o resultado em um `.json`.

## ⚙️ Como Usar

### 1. Instalar Dependências

Este projeto usa o `requirements.txt` para gerenciar as dependências.

```bash
pip install -r requirements.txt
```
### 2. Rodar o Pipeline de Treino (Default)

O pipeline `__default__` (padrão) executará o pré-processamento e a modelagem. Isso irá gerar todos os artefatos necessários (modelo, vetorizador, relatórios).

```bash
kedro run
```
(Nota: O sample_frac pode ser ajustado em conf/base/parameters.yml para treinar com mais ou menos dados.)
### 3. Rodar o Pipeline de Produção (Inferência)

Após o pipeline de treino ter sido executado pelo menos uma vez, você pode usar o pipeline de produção para prever novos dados.

1.  **Crie seu arquivo de entrada:** Adicione os novos clientes (sem a coluna `nota`) ao arquivo `data/01_raw/cliente_para_prever.json`. O formato deve ser uma lista de JSONs, similar ao arquivo de treino original.

2.  **Execute o pipeline `producao`:**

    ```bash
    kedro run --pipeline=producao
    ```

3.  **Verifique o resultado:** A previsão será salva no arquivo `data/08_reporting/previsao_final.json`.

---

## Testes

O projeto inclui um conjunto de testes básicos. Para executá-los:

```bash
pytest
```

## Package your Kedro project
[Para mais informações sobre como o Kedro funciona, confira a](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
