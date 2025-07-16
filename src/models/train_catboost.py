#!/usr/bin/env python
"""
Treinamento e forecast CatBoost passo a passo (rolling forecast) em pandas.
"""
import logging
from pathlib import Path
import argparse
import pandas as pd
import joblib
import math
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor


def evaluate(y_true, y_pred):
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = math.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = (abs((y_true - y_pred) / y_true.replace(0, 1)).mean()) * 100
    wmape = abs(y_true - y_pred).sum() / y_true.sum() * 100
    return dict(rmse=rmse, mae=mae, r2=r2, mape=mape, wmape=wmape)


def generate_all_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 7, 14],
    windows: list[int] = [7, 14, 30]
) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat['was_imputed'] = df_feat.get('is_imputed', False).astype(int)

    # lags
    for lag in lags:
        df_feat[f'lag_{lag}'] = (
            df_feat.groupby('sku')['venda']
                   .shift(lag)
                   .fillna(0)
        )

    # rolling means
    for w in windows:
        df_feat[f'roll_mean_{w}'] = (
            df_feat.groupby('sku')['venda']
                   .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
                   .fillna(0)
        )

    # calendar
    df_feat['data'] = pd.to_datetime(df_feat['data'])
    df_feat['weekday']    = df_feat['data'].dt.weekday
    df_feat['is_weekend'] = df_feat['weekday'].isin([5, 6]).astype(int)
    df_feat['month']      = df_feat['data'].dt.month

    return df_feat


class CBTrainer:
    def __init__(self, features_csv, model_out, forecast_out, log_path, folds=5, horizon=7):
        self.features_csv = features_csv
        self.model_out    = Path(model_out)
        self.forecast_out = Path(forecast_out)
        self.log_path     = log_path
        self.folds        = folds
        self.horizon      = horizon
        self.feature_cols: list[str] = []
        self._setup_logging()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("pipeline").info("==== INÍCIO CATBOOST ====")

    def load_data(self) -> pd.DataFrame:
        logger = logging.getLogger("pipeline")
        p = Path(self.features_csv)
        csv = next(p.glob("*.csv")) if p.is_dir() else p
        df_raw = pd.read_csv(csv, parse_dates=['data'])
        logger.info(f"Dados brutos carregados: {df_raw.shape}")

        df = generate_all_features(df_raw)
        logger.info(f"Features geradas: {df.shape}")

        # fixa lista de colunas de feature
        self.feature_cols = [c for c in df.columns if c not in ('sku', 'data', 'venda')]
        logger.info(f"Colunas de feature: {self.feature_cols}")

        return df

    def cross_validate(self, df: pd.DataFrame):
        logger = logging.getLogger("pipeline")
        X = df[self.feature_cols]
        y = df['venda']
        tscv = TimeSeriesSplit(n_splits=self.folds)
        rmses = []
        for i, (tr, vl) in enumerate(tscv.split(X), 1):
            model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                verbose=False,
                random_state=42
            )
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[vl])
            m = evaluate(y.iloc[vl], pred)
            rmses.append(m['rmse'])
            logger.info(f"fold {i} RMSE {m['rmse']:.4f}")
        logger.info(f"CV RMSE média {pd.Series(rmses).mean():.4f}")

    def train_and_save(self, df: pd.DataFrame):
        logger = logging.getLogger("pipeline")
        X = df[self.feature_cols]
        y = df['venda']
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            verbose=False,
            random_state=42
        )
        model.fit(X, y)
        self.model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(self.model_out))
        logger.info(f"Modelo salvo em {self.model_out}")
        return model

    def rolling_forecast(self, model, df: pd.DataFrame) -> pd.DataFrame:
        logger = logging.getLogger("pipeline")
        df_work = df.copy()
        preds = []

        for step in range(1, self.horizon + 1):
            df_feat = generate_all_features(df_work)
            last_date = df_feat['data'].max()
            df_last = df_feat[df_feat['data'] == last_date]

            Xf = df_last[self.feature_cols]
            yhat = model.predict(Xf)

            df_next = df_last.copy()
            df_next['data']  = last_date + pd.Timedelta(days=1)
            df_next['venda'] = yhat

            df_work = pd.concat([df_work, df_next], ignore_index=True)
            preds.append(
                df_next[['sku','data','venda']]
                .rename(columns={'venda':'cb_pred'})
            )
            logger.info(f"Previsto dia {df_next['data'].iloc[0].date()}")
        return pd.concat(preds, ignore_index=True)

    def run(self):
        logger = logging.getLogger("pipeline")
        df = self.load_data()
        self.cross_validate(df)
        model = self.train_and_save(df)

        # in-sample
        df_ins = df[['sku','data']].copy()
        df_ins['cb_pred'] = model.predict(df[self.feature_cols])

        # futuro recursivo
        df_fut = self.rolling_forecast(model, df)

        # concat e salvar
        out = pd.concat([df_ins, df_fut], ignore_index=True) \
                .sort_values(['sku','data'])
        self.forecast_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.forecast_out, index=False)
        logger.info(f"Forecast salvo em {self.forecast_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="data/features/vendas_features.csv")
    parser.add_argument("--model-out",    default="data/models/catboost_model.cbm")
    parser.add_argument("--forecast-out", default="data/models/catboost_forecast.csv")
    parser.add_argument("--log",          default="data/logs/pipeline.log")
    parser.add_argument("--folds",   type=int, default=5)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    CBTrainer(
        features_csv=args.features_csv,
        model_out=args.model_out,
        forecast_out=args.forecast_out,
        log_path=args.log,
        folds=args.folds,
        horizon=args.horizon
    ).run()

if __name__ == "__main__":
    main()
