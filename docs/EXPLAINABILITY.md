# Explicabilidade no C1D ForecastLab

## Objetivo

Descrever quais atributos contribuíram para as previsões dos modelos LightGBM
e CatBoost, sem interpretar atribuição preditiva como efeito causal.

## Artefatos

Após treinar os dois modelos, execute:

```bash
forecast-explain \
  --features-csv data/features/vendas_features.csv \
  --lgbm-model data/models/lgbm_model.pkl \
  --catboost-model data/models/catboost_model.cbm \
  --output-dir data/explainability \
  --horizon 7
```

O processo produz:

- `shap_global.csv`: importância agregada por modelo e atributo;
- `shap_local.parquet` ou `shap_local.csv`: contribuições por SKU, data e
  horizonte;
- `shap_validation.csv`: resíduo máximo da validação de aditividade;
- `manifest.json`: contrato dos artefatos gerados.

## Definições

Para uma previsão individual:

$$
\hat{y}_i = \phi_0 + \sum_{j=1}^{p}\phi_{ij}
$$

em que $\phi_0$ é o valor-base e $\phi_{ij}$ é a contribuição SHAP do
atributo $j$ para a previsão $i$.

A importância global exibida no dashboard é:

$$
I_j = \frac{1}{n}\sum_{i=1}^{n}|\phi_{ij}|
$$

O valor absoluto impede que contribuições positivas e negativas se cancelem.

## Validade e limitações

- O cálculo usa as implementações SHAP nativas dos modelos de árvores.
- A soma do valor-base e das contribuições é comparada à previsão direta.
- A amostra global privilegia observações recentes e tem tamanho configurável.
- A explicação local do forecast respeita os atributos de cada horizonte
  recursivo.
- Por padrão, o artefato local futuro cobre os 500 SKUs com maior venda nos
  30 dias recentes. O limite pode ser alterado com `--max-future-skus`.
- Uma contribuição positiva indica aumento em relação ao valor-base do modelo,
  não o efeito de uma intervenção.
- Atributos correlacionados podem compartilhar ou redistribuir importância.
- Uma explicação coerente não compensa desempenho preditivo insuficiente.
