"""C1D ForecastLab — dashboard editorial de forecasting."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

# O Streamlit Cloud executa este arquivo diretamente e adiciona apenas
# src/visualization ao sys.path. Incluímos src para tornar os pacotes do
# projeto importáveis sem depender de uma instalação editável.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from eda.demand_profile import (
    ADI_THRESHOLD,
    CATEGORY_ORDER,
    CV2_THRESHOLD,
    classify_demand_profiles,
)


PAPER = "#f1ebdc"
INK = "#171612"
RED = "#a43a2b"
YELLOW = "#d3a329"
GREY = "#656057"
LINE = "rgba(23, 22, 18, .28)"
CATEGORY_COLORS = {
    "Regular": "#d3a329",
    "Errática": "#c77f50",
    "Intermitente": "#a43a2b",
    "Irregular": "#171612",
    "Sem demanda": "#8b857a",
}

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


def chart_header(title: str, explanation: str):
    """Título compacto com balão de ajuda metodológica."""
    label, help_column = st.columns([0.94, 0.06], vertical_alignment="center")
    label.markdown(f"#### {title}")
    with help_column:
        with st.popover("ⓘ", help=f"Como interpretar {title}"):
            st.markdown(explanation)


@st.cache_data
def load_csv(path: str, date_col: str | None = None) -> pd.DataFrame:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    file = next(candidate.glob("*.csv")) if candidate.is_dir() else candidate
    kwargs = {"parse_dates": [date_col]} if date_col else {}
    return pd.read_csv(file, **kwargs)


@st.cache_data
def load_local_shap(cache_version: str = "v2-project-root") -> pd.DataFrame:
    """Carrega SHAP local pela raiz do projeto e invalida caches pré-artefato."""
    del cache_version
    parquet = PROJECT_ROOT / "data/explainability/shap_local.parquet"
    csv = PROJECT_ROOT / "data/explainability/shap_local.csv"
    if parquet.is_file() and parquet.stat().st_size > 0:
        frame = pd.read_parquet(parquet)
    elif csv.is_file() and csv.stat().st_size > 0:
        frame = pd.read_csv(csv, parse_dates=["data"])
    else:
        return pd.DataFrame()
    if "data" in frame:
        frame["data"] = pd.to_datetime(frame["data"])
    return frame


def existing_data(path: str, date_col: str | None = None) -> pd.DataFrame:
    try:
        return load_csv(path, date_col)
    except (FileNotFoundError, StopIteration, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data
def compute_demand_profiles(sales: pd.DataFrame) -> pd.DataFrame:
    return classify_demand_profiles(sales)


def filter_skus(frame: pd.DataFrame, skus: set[str]) -> pd.DataFrame:
    if frame.empty or "sku" not in frame:
        return frame
    return frame.loc[frame["sku"].astype(str).isin(skus)].copy()


def aggregate_daily(sku_data: pd.DataFrame) -> pd.DataFrame:
    if sku_data.empty:
        return pd.DataFrame()
    aggregations = {}
    for source, target in (
        ("realized", "realized_total"),
        ("lgbm_pred", "lgbm_total"),
        ("cb_pred", "catboost_total"),
        ("reconciled", "reconciled_total"),
    ):
        if source in sku_data:
            aggregations[target] = (source, "sum")
    if not aggregations:
        return pd.DataFrame()
    return sku_data.groupby("data").agg(**aggregations).reset_index()


def metrics_for_skus(sku_data: pd.DataFrame, historical_end) -> pd.DataFrame:
    required = {"data", "realized", "lgbm_pred", "cb_pred"}
    if sku_data.empty or not required.issubset(sku_data):
        return pd.DataFrame()
    frame = sku_data.loc[pd.to_datetime(sku_data["data"]).le(historical_end)].copy()
    frame["reconciled"] = frame[["lgbm_pred", "cb_pred"]].mean(axis=1)

    def measures(y_true, y_pred):
        y_true = pd.to_numeric(y_true, errors="coerce")
        y_pred = pd.to_numeric(y_pred, errors="coerce")
        valid = y_true.notna() & y_pred.notna()
        y_true, y_pred = y_true[valid], y_pred[valid]
        if y_true.empty:
            return {}
        error = y_true - y_pred
        denominator = y_true.abs().sum()
        variance = ((y_true - y_true.mean()) ** 2).sum()
        return {
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mape": float(np.mean(np.abs(error) / y_true.abs().replace(0, 1)) * 100),
            "wmape": float(np.abs(error).sum() / denominator * 100) if denominator else np.nan,
            "r2": float(1 - (error ** 2).sum() / variance) if variance else np.nan,
        }

    records = []
    for name, column in (
        ("lgbm", "lgbm_pred"),
        ("catboost", "cb_pred"),
        ("reconciled", "reconciled"),
    ):
        records.append(
            {"model": name, "level": "granular", **measures(frame["realized"], frame[column])}
        )
        aggregate = frame.groupby("data")[["realized", column]].sum()
        records.append(
            {"model": name, "level": "aggregate", **measures(aggregate["realized"], aggregate[column])}
        )
    return pd.DataFrame(records)


def global_shap_for_skus(local_shap: pd.DataFrame) -> pd.DataFrame:
    if local_shap.empty:
        return pd.DataFrame()
    frame = local_shap.loc[local_shap["scope"].eq("historical")].copy()
    if frame.empty:
        frame = local_shap.copy()
    summary = (
        frame.groupby(["model", "feature"], as_index=False)
        .agg(
            mean_abs_shap=("abs_shap", "mean"),
            mean_shap=("shap_value", "mean"),
            positive_share=("shap_value", lambda values: values.gt(0).mean()),
            sample_rows=("shap_value", "size"),
            mean_base_value=("base_value", "mean"),
        )
    )
    summary["rank"] = summary.groupby("model")["mean_abs_shap"].rank(
        method="first", ascending=False
    )
    return summary


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



def eda_view(sales: pd.DataFrame, profiles: pd.DataFrame):
    page_title(
        "EDA · Perfil da demanda",
        "Frequência, variabilidade e comportamento das vendas por SKU.",
        "02",
    )
    if sales.empty or profiles.empty:
        st.warning("Execute `forecast-clean` e `forecast-eda` para gerar a análise.")
        return

    minimum = pd.to_datetime(sales["data"]).min().date()
    maximum = pd.to_datetime(sales["data"]).max().date()
    selected = st.sidebar.date_input(
        "Período da EDA",
        value=[minimum, maximum],
        min_value=minimum,
        max_value=maximum,
        key="eda_period",
    )
    filtered = sales.copy()
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        filtered = filtered.loc[
            pd.to_datetime(filtered["data"]).dt.date.between(selected[0], selected[1])
        ]

    positive = filtered.loc[filtered["venda"].gt(0)]
    day_count = filtered["data"].nunique()
    active_per_day = positive.groupby("data")["sku"].nunique()
    metrics = st.columns(5)
    metrics[0].metric(
        "SKUs no recorte",
        f"{filtered['sku'].nunique():,}",
        help="Quantidade de SKUs pertencentes às categorias selecionadas e presentes no período.",
    )
    metrics[1].metric(
        "Venda total",
        f"{filtered['venda'].sum():,.0f}",
        help="Soma da variável venda para os SKUs e datas selecionados.",
    )
    metrics[2].metric(
        "SKUs vendidos/dia",
        f"{active_per_day.mean():,.0f}" if not active_per_day.empty else "—",
        help="Média diária de SKUs com venda estritamente maior que zero.",
    )
    zero_share = filtered["venda"].le(0).mean() if len(filtered) else np.nan
    metrics[3].metric(
        "Registros sem venda",
        f"{zero_share:.1%}" if pd.notna(zero_share) else "—",
        help="Proporção de combinações SKU × data cuja venda foi zero.",
    )
    metrics[4].metric(
        "Dias analisados",
        f"{day_count:,}",
        help="Número de datas distintas incluídas no período selecionado.",
    )

    daily_sales = (
        filtered.groupby("data", as_index=False)
        .agg(vendas=("venda", "sum"), skus_ativos=("sku", lambda values: values[filtered.loc[values.index, "venda"].gt(0)].nunique()))
        .sort_values("data")
    )
    daily_sales["media_movel_7d"] = daily_sales["vendas"].rolling(7, min_periods=1).mean()

    left, right = st.columns([1.7, 1])
    with left:
        chart_header(
            "Comportamento das vendas no período",
            "A linha diária mostra o volume observado em cada data. A média móvel de 7 dias "
            "reduz oscilações pontuais e evidencia o nível recente da demanda; ela é descritiva "
            "e não representa uma previsão.",
        )
        temporal = daily_sales.melt(
            "data",
            value_vars=["vendas", "media_movel_7d"],
            var_name="Série",
            value_name="Vendas",
        )
        temporal["Série"] = temporal["Série"].map(
            {"vendas": "Venda diária", "media_movel_7d": "Média móvel 7d"}
        )
        chart = (
            alt.Chart(temporal)
            .mark_line(point=alt.OverlayMarkDef(size=15), strokeWidth=2)
            .encode(
                x=alt.X("data:T", title="DATA"),
                y=alt.Y("Vendas:Q", title="VENDAS"),
                color=alt.Color(
                    "Série:N",
                    scale=alt.Scale(domain=["Venda diária", "Média móvel 7d"], range=[RED, INK]),
                    title=None,
                ),
                tooltip=["data:T", "Série:N", alt.Tooltip("Vendas:Q", format=",.0f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(style_chart(chart), use_container_width=True)
    with right:
        chart_header(
            "SKUs com venda por dia",
            "Conta quantos SKUs tiveram venda maior que zero em cada data. O indicador mede "
            "amplitude do sortimento ativo, não o volume vendido.",
        )
        active = (
            alt.Chart(daily_sales)
            .mark_area(line={"color": YELLOW}, color=YELLOW, opacity=0.28)
            .encode(
                x=alt.X("data:T", title="DATA"),
                y=alt.Y("skus_ativos:Q", title="SKUS ATIVOS"),
                tooltip=["data:T", alt.Tooltip("skus_ativos:Q", title="SKUs")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(style_chart(active), use_container_width=True)

    st.subheader("Classificação ADI × CV²")
    counts = (
        profiles.groupby("categoria_demanda", observed=False)
        .size()
        .rename("skus")
        .reset_index()
    )
    counts = counts.loc[counts["skus"].gt(0)]
    counts["participacao"] = counts["skus"] / counts["skus"].sum()
    domain = [category for category in CATEGORY_ORDER if category in counts["categoria_demanda"].astype(str).tolist()]
    colors = [CATEGORY_COLORS[category] for category in domain]

    left, right = st.columns([1, 1.5])
    with left:
        chart_header(
            "Distribuição dos perfis",
            "Mostra quantos SKUs pertencem a cada padrão ADI × CV². As categorias são "
            "calculadas no histórico completo e permanecem fixas durante os filtros de período.",
        )
        bars = (
            alt.Chart(counts)
            .mark_bar(size=25)
            .encode(
                y=alt.Y("categoria_demanda:N", sort=domain, title=None),
                x=alt.X("skus:Q", title="SKUS"),
                color=alt.Color(
                    "categoria_demanda:N",
                    scale=alt.Scale(domain=domain, range=colors),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("categoria_demanda:N", title="Categoria"),
                    alt.Tooltip("skus:Q", title="SKUs"),
                    alt.Tooltip("participacao:Q", title="Participação", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(style_chart(bars), use_container_width=True)
    with right:
        chart_header(
            "Mapa de frequência × variabilidade",
            "**Eixo X — ADI:** intervalo médio entre dias com venda; quanto mais à direita, "
            "mais espaçada é a demanda.\n\n**Eixo Y — CV²:** variabilidade do volume nos dias "
            "em que houve venda; quanto mais acima, mais instável é o tamanho da demanda.",
        )
        scatter_data = profiles.loc[
            np.isfinite(profiles["adi"]) & profiles["cv2"].notna()
        ].copy()
        if not scatter_data.empty:
            adi_cap = max(ADI_THRESHOLD * 1.1, scatter_data["adi"].quantile(0.99))
            cv2_cap = max(CV2_THRESHOLD * 1.1, scatter_data["cv2"].quantile(0.99))
            scatter_data["adi_plot"] = scatter_data["adi"].clip(upper=adi_cap)
            scatter_data["cv2_plot"] = scatter_data["cv2"].clip(upper=cv2_cap)
            points = (
                alt.Chart(scatter_data)
                .mark_circle(size=48, opacity=0.58)
                .encode(
                    x=alt.X("adi_plot:Q", title="ADI · INTERVALO MÉDIO"),
                    y=alt.Y("cv2_plot:Q", title="CV² · VARIABILIDADE"),
                    color=alt.Color(
                        "categoria_demanda:N",
                        scale=alt.Scale(domain=domain, range=colors),
                        title=None,
                    ),
                    tooltip=[
                        "sku:N",
                        alt.Tooltip("categoria_demanda:N", title="Categoria"),
                        alt.Tooltip("adi:Q", format=".2f"),
                        alt.Tooltip("cv2:Q", format=".2f"),
                        alt.Tooltip("venda_total:Q", format=",.0f", title="Venda total"),
                    ],
                )
            )
            vline = alt.Chart(pd.DataFrame({"x": [ADI_THRESHOLD]})).mark_rule(
                color=INK, strokeDash=[5, 4]
            ).encode(x="x:Q")
            hline = alt.Chart(pd.DataFrame({"y": [CV2_THRESHOLD]})).mark_rule(
                color=INK, strokeDash=[5, 4]
            ).encode(y="y:Q")
            chart = (points + vline + hline).properties(height=300).interactive()
            st.altair_chart(style_chart(chart), use_container_width=True)

    with st.expander("ⓘ Como interpretar os quadrantes do mapa ADI × CV²"):
        st.markdown(
            """
            As linhas tracejadas representam os limites metodológicos **ADI = 1,32**
            e **CV² = 0,49**.

            | Região do mapa | Categoria | Racional |
            |---|---|---|
            | Inferior esquerda | **Regular** | Venda frequente e volume relativamente estável. |
            | Superior esquerda | **Errática** | Venda frequente, porém com grande variação de volume. |
            | Inferior direita | **Intermitente** | Muitos intervalos sem venda, mas volume relativamente estável quando ocorre. |
            | Superior direita | **Irregular** | Venda espaçada e volume variável; é o padrão mais difícil de prever. |

            Cada ponto representa um SKU. Pontos mais à direita apresentam vendas menos
            frequentes; pontos mais altos possuem volumes mais voláteis. A classificação
            descreve o histórico e não determina causalidade nem, isoladamente, o melhor modelo.
            """
        )
    st.caption(
        "ADI e CV² são calculados no histórico completo até a data de treinamento. "
        "O período altera os indicadores temporais, mas não redefine a categoria."
    )
    table = profiles.copy()
    table["percentual_dias_sem_venda"] = table["percentual_dias_sem_venda"].map(
        lambda value: f"{value:.1%}"
    )
    st.dataframe(
        table[
            [
                "sku", "categoria_demanda", "adi", "cv2", "dias_observados",
                "dias_com_venda", "percentual_dias_sem_venda", "venda_total",
                "venda_media_quando_ocorre",
            ]
        ].rename(
            columns={
                "sku": "SKU",
                "categoria_demanda": "Categoria",
                "adi": "ADI",
                "cv2": "CV²",
                "dias_observados": "Dias observados",
                "dias_com_venda": "Dias com venda",
                "percentual_dias_sem_venda": "% dias sem venda",
                "venda_total": "Venda total",
                "venda_media_quando_ocorre": "Venda média por ocorrência",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

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

        ## Exploração e perfis de demanda
        ADI e CV² descrevem frequência e variabilidade por SKU. A segmentação é
        exploratória, permanece fixa no histórico de treinamento e não implica causalidade.

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

    original_daily = existing_data("data/models/reconciled_daily_summary.csv", "data")
    sku_data = existing_data("data/models/reconciled_sku_forecast.csv", "data")
    original_metrics = existing_data("data/models/metrics_report.csv")
    global_shap = existing_data("data/explainability/shap_global.csv")
    local_shap = load_local_shap()
    sales = existing_data("data/processed/vendas_processed.csv", "data")
    profiles = existing_data("data/eda/sku_demand_profile.csv")

    if profiles.empty and not sales.empty:
        profiles = compute_demand_profiles(sales)
    if not profiles.empty:
        profiles["categoria_demanda"] = profiles["categoria_demanda"].astype(str)
        available_categories = [
            category
            for category in CATEGORY_ORDER
            if category in profiles["categoria_demanda"].unique()
        ]
        selected_categories = st.sidebar.multiselect(
            "Categoria de demanda",
            available_categories,
            default=available_categories,
            help="Classificação fixa pelo histórico completo: ADI × CV².",
        )
        if not selected_categories:
            st.sidebar.info("Nenhuma categoria selecionada: exibindo todas.")
            selected_categories = available_categories
        selected_skus = set(
            profiles.loc[
                profiles["categoria_demanda"].isin(selected_categories), "sku"
            ].astype(str)
        )
        profiles = profiles.loc[
            profiles["categoria_demanda"].isin(selected_categories)
        ].copy()
        sales = filter_skus(sales, selected_skus)
        sku_data = filter_skus(sku_data, selected_skus)
        local_shap = filter_skus(local_shap, selected_skus)

        all_selected = set(selected_categories) == set(available_categories)
        if not all_selected:
            global_filtered = global_shap_for_skus(local_shap)
            if not global_filtered.empty:
                global_shap = global_filtered
    else:
        st.sidebar.caption("Perfil ADI/CV² ainda não disponível.")

    daily = aggregate_daily(sku_data)
    if daily.empty:
        daily = original_daily

    historical_end = (
        pd.to_datetime(sales["data"]).max()
        if not sales.empty
        else pd.Timestamp(datetime.date.today())
    )
    metrics = metrics_for_skus(sku_data, historical_end)
    if metrics.empty:
        metrics = original_metrics

    navigation = {
        "01 EDA · Perfil da demanda": lambda: eda_view(sales, profiles),
        "02 Visão executiva": lambda: executive_view(
            daily, sku_data, metrics, global_shap
        ),
        "03 Desempenho temporal": lambda: temporal_view(daily),
        "04 Explorador de SKU": lambda: sku_view(sku_data, local_shap),
        "05 Modelos": lambda: models_view(metrics, global_shap),
        "06 Método": method_view,
    }
    selected = st.sidebar.radio("Navegação", list(navigation))
    navigation[selected]()
    footer()


if __name__ == "__main__":
    main()
