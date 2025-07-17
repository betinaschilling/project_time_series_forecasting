import sys
import datetime
from pathlib import Path

import streamlit as st
import streamlit.web.cli as stcli
import pandas as pd
import altair as alt

# --- Configurações da página ---
st.set_page_config(
    page_title="Dashboard de Forecast",
    layout="wide",
)

# --- Função genérica de CSV com cache ---
@st.cache_data
def load_csv(path: str, date_col: str = None) -> pd.DataFrame:
    p = Path(path)
    f = next(p.glob("*.csv")) if p.is_dir() else p
    if date_col:
        return pd.read_csv(f, parse_dates=[date_col])
    return pd.read_csv(f)

# --- Título geral ---
st.title("Forecast de Vendas")

# --- Menu lateral ---
view = st.sidebar.radio(
    "Selecione a visão",
    ["Resumo Aplicação", "Resumo Diário", "Por SKU", "Métricas"]
)

# --- Aba: Storytelling ---
if view == "Resumo Aplicação":
    st.header("Análise Geral do Projeto")
    st.markdown("""
        **Problema**  
        - Histórico diário de vendas por SKU, mas demanda futura incerta.  
        - SKUs novos sem dados históricos; sazonalidade e comportamento diversificado.  
        - Necessidade de forecasts granulares e agregados, com reconciliação para consistência.

        **Solução**  
        1. **Ingestão & Pré-processamento**  
           - Leitura do CSV bruto; expansão de datas (zero-fill); limpeza de outliers.  
        2. **Feature Engineering**  
           - Geração de lags, médias móveis, variáveis de calendário e flags de imputação.  
        3. **Modelagem**  
           - Treino de LightGBM e CatBoost no nível SKU; previsão granular e agregada.  
        4. **Reconciliação**  
           - Combinação (média) de LGBM + CatBoost ajustada por quota histórica.  
        5. **Avaliação**  
           - Cálculo de RMSE, MAE, MAPE, WMAPE e R² em granular e agregado.  
        6. **Visualização**  
           - Dashboard em Streamlit: filtros, formatação e storytelling integrado. 

        Github: https://github.com/betinaschilling/project_time_series_forecasting 
    """)

# --- Aba: Resumo Diário ---
elif view == "Resumo Diário":
    st.header("Resumo Diário Agregado")

    # filtro de período
    start_date, end_date = st.sidebar.date_input(
        "Período",
        value=[datetime.date(2024, 1, 1), datetime.date(2024, 4, 15)],
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2024, 12, 31)
    )

    # carrega e filtra
    df = load_csv("data/models/reconciled_daily_summary.csv", date_col="data")
    df = df[(df["data"].dt.date >= start_date) & (df["data"].dt.date <= end_date)].copy()

    # renomeia
    df = df.rename(columns={
        "data":             "Data",
        "realized_total":   "Realizado",
        "lgbm_total":      "Forecast LGBM",
        "catboost_total":   "Forecast CatBoost",
    })

    # formata
    df["Data"] = df["Data"].dt.date
    for col in ["Realizado", "Forecast LGBM", "Forecast CatBoost"]:
        df[col] = df[col].round(0).astype(int).map(lambda x: f"{x:,}")

    # exibe tabela
    st.dataframe(df[["Data", "Realizado", "Forecast CatBoost", "Forecast LGBM"]], width=1200, height=300)

    # gráfico: só Realizado x CatBoost
    df_plot = df.copy()
    for col in ["Realizado", "Forecast CatBoost", "Forecast LGBM"]:
        df_plot[col] = df_plot[col].str.replace(",", "").astype(int)

    df_melt = df_plot.melt(
        id_vars=["Data"],
        value_vars=["Realizado", "Forecast LGBM", "Forecast CatBoost"],
        var_name="SÉRIE",
        value_name="VENDAS"
    )

    chart = (
        alt.Chart(df_melt)
        .mark_line(point=True)
        .encode(
            x=alt.X("Data:T", title="DATA"),
            y=alt.Y("VENDAS:Q", title="VENDAS"),
            color=alt.Color("SÉRIE:N", title=None)
        )
        .properties(
            title="FORECAST vs REALIZADO POR DIA",
            width="container",
            height=400
        )
    )
    st.altair_chart(chart, use_container_width=True)

# --- Aba: Por SKU ---
elif view == "Por SKU":
    st.header("Forecast por SKU")
    df = load_csv("data/models/reconciled_sku_forecast.csv", date_col="data")
    skus = sorted(df["sku"].unique())
    sku = st.sidebar.selectbox("Escolha o SKU", skus)
    df = df[df["sku"] == sku].copy()

    # renomeia
    df = df.rename(columns={
        "sku":       "Cód SKU",
        "data":      "Data",
        "cb_pred":   "Forecast CatBoost",
        "lgbm_pred": "Forecast LGBM",
        "realized":  "Realizado",
    })
    df["Data"] = df["Data"].dt.date

    # move Realizado para 2º coluna
    df = df[["Cód SKU", "Data", "Realizado", "Forecast LGBM", "Forecast CatBoost", "reconciled"]]

    # formata
    for col in ["Realizado", "Forecast LGBM", "Forecast CatBoost"]:
        df[col] = df[col].round(0).astype(int).map(lambda x: f"{x:,}")

    st.dataframe(df.drop(columns=["reconciled"]), width=1200, height=300)

    # gráfico: só Realizado x CatBoost
    df_plot = df.copy()
    for col in ["Realizado", "Forecast CatBoost"]:
        df_plot[col] = df_plot[col].str.replace(",", "").astype(int)
    df_melt = df_plot.melt(
        id_vars=["Data"],
        value_vars=["Realizado", "Forecast CatBoost"],
        var_name="SÉRIE",
        value_name="VENDAS"
    )

    chart = (
        alt.Chart(df_melt)
        .mark_line(point=True)
        .encode(
            x=alt.X("Data:T", title="DATA"),
            y=alt.Y("VENDAS:Q", title="VENDAS"),
            color=alt.Color("SÉRIE:N", title=None)
        )
        .properties(
            title=f"FORECAST vs REALIZADO – SKU {sku}",
            width="container",
            height=400
        )
    )
    st.altair_chart(chart, use_container_width=True)

# --- Aba: Métricas ---
else:
    st.header("Avaliação de Métricas")
    df = load_csv("data/models/metrics_report.csv")

    # filtro de nível, default = granular
    levels = df["level"].unique().tolist()
    idx = levels.index("granular") if "granular" in levels else 0
    lvl = st.sidebar.selectbox("Selecione o nível", ["Todos"] + levels, index=idx+1)
    if lvl != "Todos":
        df = df[df["level"] == lvl]

    # renomeia e uppercase
    df = df.rename(columns={
        "model":  "MODELO",
        "level":  "NÍVEL",
        "rmse":   "RMSE",
        "mae":    "MAE",
        "mape":   "MAPE (%)",
        "wmape":  "WMAPE (%)",
        "r2":     "R²"
    })
    for col in ["RMSE","MAE","MAPE (%)","WMAPE (%)","R²"]:
        df[col] = df[col].round(2).map(lambda x: f"{x:,}")

    st.dataframe(df, width=1200, height=300)

    # gráfico de barras horizontal
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("MODELO:N", title="MODELO"),
            x=alt.X("RMSE:Q", title="RMSE"),
            color=alt.Color("MODELO:N", legend=None),
            tooltip=["MAE","MAPE (%)","WMAPE (%)","R²"]
        )
        .properties(
            title="COMPARAÇÃO DE RMSE POR MODELO",
            width="container",
            height=400
        )
    )
    # adiciona rótulo de valor em cada barra
    chart = chart + chart.mark_text(
        align="left",
        dx=3
    ).encode(
        text=alt.Text("RMSE:Q", format=".2f")
    )
    st.altair_chart(chart, use_container_width=True)

