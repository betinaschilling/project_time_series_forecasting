"""C1D ForecastLab — dashboard editorial de forecasting."""

from __future__ import annotations

import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PAPER = "#f1ebdc"
INK = "#171612"
RED = "#a43a2b"
YELLOW = "#d3a329"
GREY = "#656057"
LINE = "rgba(23, 22, 18, .28)"

FEATURE_LABELS = {
    "lag_1": "Venda D−1",
    "lag_7": "Venda D−7",
    "lag_14": "Venda D−14",
    "roll_mean_7": "Média móvel 7d",
    "roll_mean_14": "Média móvel 14d",
    "roll_mean_30": "Média móvel 30d",
    "weekday": "Dia da semana",
    "is_weekend": "Fim de semana",
    "month": "Mês",
    "was_imputed": "Dado imputado",
    "is_imputed": "Flag de imputação",
}


def inject_style():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {{
            --paper: {PAPER}; --ink: {INK}; --red: {RED};
            --yellow: {YELLOW}; --grey: {GREY}; --line: {LINE};
        }}
        .stApp, [data-testid="stSidebar"] {{ background: var(--paper); color: var(--ink); }}
        .stApp {{
            background-image: radial-gradient(rgba(23,22,18,.025) .7px, transparent .7px);
            background-size: 5px 5px;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid var(--line); min-width: 255px; max-width: 255px;
        }}
        [data-testid="stSidebar"] * {{ font-family: Arial, sans-serif; }}
        .block-container {{ padding: 1.5rem 2rem 2rem; max-width: 1500px; }}
        h1, h2, h3 {{ font-family: Georgia, 'Times New Roman', serif !important; color: var(--ink); }}
        h1 {{ font-size: clamp(2.7rem, 5vw, 5rem) !important; font-weight: 400 !important;
             letter-spacing: -.045em !important; line-height: .96 !important; }}
        h2 {{ font-weight: 400 !important; border-top: 1px solid var(--line);
             padding-top: .7rem; }}
        p, label, [data-testid="stMetricLabel"] {{ color: var(--ink); }}
        [data-testid="stMetric"] {{
            border-top: 1px solid var(--ink); border-right: 1px solid var(--line);
            padding: .75rem 1rem .45rem 0;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;
            letter-spacing: .035em; font-size: .72rem;
        }}
        [data-testid="stMetricValue"] {{
            font-family: Georgia, serif; font-size: 2rem; color: var(--ink);
        }}
        .masthead {{
            display:flex; align-items:baseline; justify-content:space-between;
            border-bottom:1px solid var(--ink); padding-bottom:.65rem; margin-bottom:1.4rem;
        }}
        .brand {{ display:flex; align-items:baseline; gap:.75rem; }}
        .brand-mark {{ font: 400 2.5rem/1 Georgia, serif; letter-spacing:-.04em; }}
        .brand-separator {{ color:var(--red); font:500 1.4rem 'IBM Plex Mono', monospace; }}
        .brand-name {{ color:var(--red); font:500 1rem 'IBM Plex Mono', monospace; }}
        .mast-meta {{ font: .73rem 'IBM Plex Mono', monospace; letter-spacing:.06em; }}
        .eyebrow {{ color:var(--red); font:500 .72rem 'IBM Plex Mono', monospace;
                    letter-spacing:.08em; text-transform:uppercase; }}
        .subtitle {{ font: 1rem 'IBM Plex Mono', monospace; color:var(--grey);
                     margin:.2rem 0 1.2rem; }}
        .method-strip {{
            border-top:1px solid var(--line); margin-top:1.2rem; padding:.85rem 0;
            font: .78rem 'IBM Plex Mono', monospace; word-spacing:.25rem;
        }}
        .note {{ background:rgba(227,207,121,.35); border-left:3px solid var(--yellow);
                 padding:.7rem .8rem; font: .78rem 'IBM Plex Mono', monospace; }}
        .stButton > button {{
            background:var(--red); color:white; border:0; border-radius:0;
            font-family:'IBM Plex Mono', monospace;
        }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
            background:rgba(255,255,255,.16); border-radius:0;
        }}
        @media (max-width: 900px) {{
            .mast-meta {{ display:none; }} .block-container {{ padding:1rem; }}
            .brand-name {{ font-size:.8rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def masthead():
    st.markdown(
        """
        <div class="masthead">
          <div class="brand">
            <span class="brand-mark">C1D</span>
            <span class="brand-separator">»</span>
            <span class="brand-name">ForecastLab</span>
          </div>
          <div class="mast-meta">LAB 03 · VAREJO · HORIZONTE 7D</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str, notebook: str):
    st.markdown(f'<div class="eyebrow">CADERNO {notebook}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


@st.cache_data
def load_csv(path: str, date_col: str | None = None) -> pd.DataFrame:
    candidate = Path(path)
    file = next(candidate.glob("*.csv")) if candidate.is_dir() else candidate
    kwargs = {"parse_dates": [date_col]} if date_col else {}
    return pd.read_csv(file, **kwargs)


@st.cache_data
def load_local_shap() -> pd.DataFrame:
    parquet = Path("data/explainability/shap_local.parquet")
    csv = Path("data/explainability/shap_local.csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["data"])
    return pd.DataFrame()


def existing_data(path: str, date_col: str | None = None) -> pd.DataFrame:
    try:
        return load_csv(path, date_col)
    except (FileNotFoundError, StopIteration, pd.errors.EmptyDataError):
        return pd.DataFrame()


def style_chart(chart: alt.Chart) -> alt.Chart:
    return chart.configure(
        background="transparent"
    ).configure_axis(
        gridColor="#d5cfc1",
        domainColor=INK,
        labelColor=INK,
        titleColor=INK,
        labelFont="IBM Plex Mono",
        titleFont="IBM Plex Mono",
    ).configure_legend(
        labelFont="IBM Plex Mono",
        titleFont="IBM Plex Mono",
        orient="top",
    ).configure_title(
        font="Georgia",
        fontSize=22,
        fontWeight="normal",
        color=INK,
        anchor="start",
    )


def forecast_chart(daily: pd.DataFrame):
    columns = {
        "realized_total": "Realizado",
        "lgbm_total": "LightGBM",
        "catboost_total": "CatBoost",
    }
    available = [column for column in columns if column in daily]
    if not available:
        st.info("Séries de forecast ainda não disponíveis.")
        return
    plot = daily[["data", *available]].rename(columns=columns).melt(
        "data", var_name="Série", value_name="Vendas"
    )
    chart = (
        alt.Chart(plot)
        .mark_line(point=alt.OverlayMarkDef(size=22), strokeWidth=2)
        .encode(
            x=alt.X("data:T", title="DATA"),
            y=alt.Y("Vendas:Q", title="VENDAS", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Série:N",
                scale=alt.Scale(
                    domain=["Realizado", "LightGBM", "CatBoost"],
                    range=[INK, RED, YELLOW],
                ),
                title=None,
            ),
            strokeDash=alt.StrokeDash(
                "Série:N",
                scale=alt.Scale(
                    domain=["Realizado", "LightGBM", "CatBoost"],
                    range=[[1, 0], [1, 0], [5, 3]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("data:T", title="Data"),
                "Série:N",
                alt.Tooltip("Vendas:Q", format=",.0f"),
            ],
        )
        .properties(height=390, title="Realizado × previsão")
        .interactive()
    )
    st.altair_chart(style_chart(chart), use_container_width=True)


def model_ranking(metrics: pd.DataFrame):
    st.subheader("Ranking de modelos")
    if metrics.empty:
        st.caption("Métricas ainda não disponíveis.")
        return
    granular = metrics.loc[metrics["level"].eq("granular")].copy()
    plot = granular if not granular.empty else metrics.copy()
    metric = "wmape" if "wmape" in plot else "rmse"
    plot[metric] = pd.to_numeric(plot[metric], errors="coerce")
    plot = plot.dropna(subset=[metric]).sort_values(metric)
    chart = (
        alt.Chart(plot)
        .mark_bar(size=18)
        .encode(
            y=alt.Y("model:N", sort=alt.SortField(metric, order="ascending"), title=None),
            x=alt.X(f"{metric}:Q", title=metric.upper()),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(range=[RED, YELLOW, GREY, INK]),
                legend=None,
            ),
            tooltip=["model:N", alt.Tooltip(f"{metric}:Q", format=".2f")],
        )
        .properties(height=220)
    )
    st.altair_chart(style_chart(chart), use_container_width=True)


def global_shap_panel(global_shap: pd.DataFrame, key: str = "global"):
    st.subheader("Drivers do forecast")
    if global_shap.empty:
        st.caption("Execute `forecast-explain` para gerar os artefatos SHAP.")
        return
    models = global_shap["model"].dropna().unique().tolist()
    model = st.selectbox("Modelo explicativo", models, key=f"{key}_model")
    top = (
        global_shap.loc[global_shap["model"].eq(model)]
        .nsmallest(8, "rank")
        .copy()
    )
    top["feature_label"] = top["feature"].map(FEATURE_LABELS).fillna(top["feature"])
    chart = (
        alt.Chart(top)
        .mark_bar(size=18, color=RED)
        .encode(
            y=alt.Y(
                "feature_label:N",
                sort=alt.SortField("mean_abs_shap", order="descending"),
                title=None,
            ),
            x=alt.X("mean_abs_shap:Q", title="MÉDIA |SHAP|"),
            tooltip=[
                alt.Tooltip("feature_label:N", title="Feature"),
                alt.Tooltip("mean_abs_shap:Q", format=".3f", title="Importância"),
                alt.Tooltip("positive_share:Q", format=".1%", title="% positivo"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(style_chart(chart), use_container_width=True)
    st.caption("Importância preditiva, não efeito causal.")


def sku_attention(sku_data: pd.DataFrame):
    st.subheader("SKUs sob atenção")
    required = {"sku", "realized", "lgbm_pred", "cb_pred"}
    if sku_data.empty or not required.issubset(sku_data.columns):
        st.caption("Forecast granular ainda não disponível.")
        return
    frame = sku_data.dropna(subset=["realized"]).copy()
    frame["forecast"] = frame[["lgbm_pred", "cb_pred"]].mean(axis=1)
    latest = frame.sort_values("data").groupby("sku", as_index=False).tail(1)
    denominator = latest["realized"].abs().replace(0, np.nan)
    latest["desvio_pct"] = (latest["forecast"] - latest["realized"]) / denominator * 100
    attention = latest.reindex(
        latest["desvio_pct"].abs().sort_values(ascending=False).index
    ).head(5)
    attention["Desvio"] = attention["desvio_pct"].map(
        lambda value: f"{value:+.1f}%"
    )
    st.dataframe(
        attention[["sku", "Desvio"]].rename(columns={"sku": "SKU"}),
        hide_index=True,
        use_container_width=True,
        height=220,
    )


def metric_summary(daily: pd.DataFrame, metrics: pd.DataFrame):
    realized = forecast = deviation = np.nan
    if not daily.empty:
        valid = daily.dropna(subset=["realized_total"])
        if not valid.empty:
            realized = valid["realized_total"].sum()
            forecast_columns = [
                column
                for column in ("lgbm_total", "catboost_total")
                if column in valid
            ]
            if forecast_columns:
                forecast_series = valid[forecast_columns].mean(axis=1)
                forecast = forecast_series.sum()
                deviation = (forecast - realized) / realized * 100 if realized else np.nan

    best_name, best_wmape = "—", np.nan
    if not metrics.empty and {"model", "wmape"}.issubset(metrics):
        ranked = metrics.copy()
        ranked["wmape"] = pd.to_numeric(ranked["wmape"], errors="coerce")
        ranked = ranked.dropna(subset=["wmape"]).sort_values("wmape")
        if not ranked.empty:
            best_name = str(ranked.iloc[0]["model"])
            best_wmape = ranked.iloc[0]["wmape"]

    columns = st.columns(5)
    columns[0].metric("Venda realizada", f"{realized:,.0f}" if pd.notna(realized) else "—")
    columns[1].metric("Forecast", f"{forecast:,.0f}" if pd.notna(forecast) else "—")
    columns[2].metric("Desvio", f"{deviation:+.1f}%" if pd.notna(deviation) else "—")
    columns[3].metric("WMAPE", f"{best_wmape:.2f}%" if pd.notna(best_wmape) else "—")
    columns[4].metric("Melhor modelo", best_name)


def executive_view(daily, sku_data, metrics, global_shap):
    page_title(
        "Forecast de vendas por SKU",
        "Demanda prevista, erro observado e sinais para decisão.",
        "03",
    )
    metric_summary(daily, metrics)
    forecast_chart(daily)
    left, middle, right = st.columns([1, 1, 1.15])
    with left:
        model_ranking(metrics)
    with middle:
        sku_attention(sku_data)
    with right:
        global_shap_panel(global_shap, "executive")


def temporal_view(daily):
    page_title(
        "Desempenho temporal",
        "O desenho no tempo revela onde o erro agregado se forma.",
        "04",
    )
    if daily.empty:
        st.warning("Resumo diário ainda não disponível.")
        return
    minimum, maximum = daily["data"].min().date(), daily["data"].max().date()
    selected = st.sidebar.date_input(
        "Período",
        value=[minimum, maximum],
        min_value=minimum,
        max_value=maximum,
    )
    filtered = daily
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        filtered = daily.loc[
            daily["data"].dt.date.between(selected[0], selected[1])
        ]
    forecast_chart(filtered)
    st.dataframe(filtered, hide_index=True, use_container_width=True)


def local_shap_panel(local_shap: pd.DataFrame, sku, selected_date):
    st.subheader("O que move esta previsão")
    if local_shap.empty:
        st.info("Explicação local ainda não gerada.")
        return
    frame = local_shap.copy()
    frame["data"] = pd.to_datetime(frame["data"])
    frame = frame.loc[
        frame["sku"].astype(str).eq(str(sku))
        & frame["data"].dt.date.eq(selected_date)
    ]
    if frame.empty:
        st.caption("Não há atribuições SHAP para esse SKU e data.")
        return
    model = st.selectbox("Modelo", frame["model"].unique(), key="local_model")
    frame = frame.loc[frame["model"].eq(model)].nlargest(10, "abs_shap").copy()
    frame["feature_label"] = frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"])
    frame["direction"] = np.where(frame["shap_value"].ge(0), "Aumenta", "Reduz")
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y(
                "feature_label:N",
                sort=alt.SortField("abs_shap", order="descending"),
                title=None,
            ),
            x=alt.X("shap_value:Q", title="CONTRIBUIÇÃO SHAP"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Aumenta", "Reduz"], range=[RED, INK]),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("feature_label:N", title="Feature"),
                alt.Tooltip("feature_value:Q", title="Valor", format=".2f"),
                alt.Tooltip("shap_value:Q", title="Contribuição", format="+.2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(style_chart(chart), use_container_width=True)
    prediction = frame["prediction"].iloc[0]
    base = frame["base_value"].iloc[0]
    strongest = frame.iloc[0]
    verb = "elevou" if strongest["shap_value"] >= 0 else "reduziu"
    st.markdown(
        f'<div class="note">A previsão de <b>{prediction:,.1f}</b> parte de '
        f'um valor-base de {base:,.1f}. O principal driver foi '
        f'<b>{strongest["feature_label"]}</b>, que {verb} a previsão em '
        f'{abs(strongest["shap_value"]):,.1f}.</div>',
        unsafe_allow_html=True,
    )


def sku_view(sku_data, local_shap):
    page_title(
        "Explorador de SKU",
        "Da previsão agregada à contribuição de cada atributo.",
        "05",
    )
    if sku_data.empty:
        st.warning("Forecast por SKU ainda não disponível.")
        return
    skus = sorted(sku_data["sku"].astype(str).unique())
    sku = st.sidebar.selectbox("SKU", skus)
    frame = sku_data.loc[sku_data["sku"].astype(str).eq(str(sku))].copy()
    available_dates = sorted(pd.to_datetime(frame["data"]).dt.date.unique())
    selected_date = st.sidebar.selectbox("Data explicada", available_dates, index=len(available_dates)-1)

    plot_columns = {
        "realized": "Realizado",
        "lgbm_pred": "LightGBM",
        "cb_pred": "CatBoost",
    }
    plot = frame[["data", *[c for c in plot_columns if c in frame]]].rename(
        columns=plot_columns
    ).melt("data", var_name="Série", value_name="Vendas")
    chart = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("data:T", title="DATA"),
            y=alt.Y("Vendas:Q", title="VENDAS", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Série:N",
                scale=alt.Scale(
                    domain=["Realizado", "LightGBM", "CatBoost"],
                    range=[INK, RED, YELLOW],
                ),
                title=None,
            ),
            tooltip=["data:T", "Série:N", alt.Tooltip("Vendas:Q", format=",.1f")],
        )
        .properties(height=360, title=f"Forecast × realizado — SKU {sku}")
    )
    st.altair_chart(style_chart(chart), use_container_width=True)
    local_shap_panel(local_shap, sku, selected_date)


def models_view(metrics, global_shap):
    page_title(
        "Avaliação dos modelos",
        "Desempenho e comportamento devem ser examinados em conjunto.",
        "06",
    )
    left, right = st.columns(2)
    with left:
        model_ranking(metrics)
    with right:
        global_shap_panel(global_shap, "models")
    if not metrics.empty:
        st.dataframe(metrics, hide_index=True, use_container_width=True)


def method_view():
    page_title(
        "Método",
        "Uma previsão só é defensável quando seu processo também é observável.",
        "07",
    )
    st.markdown(
        """
        ## Problema
        Antecipar vendas diárias por SKU em um contexto de demanda irregular.

        ## Dados confiáveis
        Malha temporal completa, identificação de imputações e atributos construídos
        sem utilizar informação futura.

        ## Modelos e inferência
        LightGBM e CatBoost com validação temporal. A previsão de sete dias é
        recursiva: cada resultado passa a integrar o histórico do horizonte seguinte.

        ## Evidências
        RMSE, MAE, MAPE, WMAPE e R² avaliam o erro. SHAP descreve como cada atributo
        contribuiu para a saída dos modelos.

        ## Decisão
        Os resultados ajudam a localizar desvios, comparar modelos e identificar
        SKUs que merecem investigação. SHAP não constitui evidência causal.
        """
    )


def footer():
    st.markdown(
        """
        <div class="method-strip">
        <span style="color:var(--red)">Problema</span> → Dados confiáveis →
        Modelos e inferência → Evidências → Decisão ↝ feedback
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="C1D » ForecastLab",
        page_icon="C1D",
        layout="wide",
    )
    inject_style()
    masthead()

    daily = existing_data("data/models/reconciled_daily_summary.csv", "data")
    sku_data = existing_data("data/models/reconciled_sku_forecast.csv", "data")
    metrics = existing_data("data/models/metrics_report.csv")
    global_shap = existing_data("data/explainability/shap_global.csv")
    local_shap = load_local_shap()

    navigation = {
        "01 Visão executiva": lambda: executive_view(
            daily, sku_data, metrics, global_shap
        ),
        "02 Desempenho temporal": lambda: temporal_view(daily),
        "03 Explorador de SKU": lambda: sku_view(sku_data, local_shap),
        "04 Modelos": lambda: models_view(metrics, global_shap),
        "05 Método": method_view,
    }
    selected = st.sidebar.radio("Navegação", list(navigation))
    navigation[selected]()
    footer()


if __name__ == "__main__":
    main()
