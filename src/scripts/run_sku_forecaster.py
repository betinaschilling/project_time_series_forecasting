#!/usr/bin/env python
# src/scripts/run_sku_forecast.py

import logging
from pathlib import Path
import argparse

import pandas as pd
import joblib

class SKUForecaster:
    def __init__(
        self,
        sku: str,
        hist_csv: str,
        features_csv: str,
        lgbm_model: str,
        cb_model: str,
        out_csv: str,
        log_path: str,
        horizon: int = 7
    ):
        self.sku           = sku
        self.hist_csv      = hist_csv
        self.features_csv  = features_csv
        self.lgbm_model    = lgbm_model
        self.cb_model      = cb_model
        self.out_csv       = Path(out_csv)
        self.log_path      = log_path
        self.horizon       = horizon
        self._setup_logging()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("pipeline").info(f"==== INÍCIO FORECAST SKU={self.sku} ====")

    def load_models(self):
        logger = logging.getLogger("pipeline")
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        logger.info(f"Carregando LGBM de {self.lgbm_model}")
        self.lgbm = joblib.load(self.lgbm_model)
        logger.info(f"Carregando CatBoost de {self.cb_model}")
        self.cb   = CatBoostRegressor().load_model(self.cb_model)

    def load_data(self):
        logger = logging.getLogger("pipeline")

        # histórico
        p_hist = Path(self.hist_csv)
        if p_hist.is_dir():
            hist_file = next(p_hist.glob("*.csv"))
        else:
            hist_file = p_hist
        hist = pd.read_csv(hist_file, parse_dates=['data'])
        hist = hist[hist['sku']==self.sku].copy()
        logger.info(f"Histórico SKU={self.sku}: {hist.shape[0]} linhas (arquivo: {hist_file})")

        # features históricos
        p_feat = Path(self.features_csv)
        if p_feat.is_dir():
            feat_file = next(p_feat.glob("*.csv"))
        else:
            feat_file = p_feat
        feats = pd.read_csv(feat_file, parse_dates=['data'])
        feats = feats[feats['sku']==self.sku].copy()
        logger.info(f"Features SKU={self.sku}: {feats.shape[0]} linhas (arquivo: {feat_file})")

        return hist, feats

    def forecast(self, hist: pd.DataFrame, feats: pd.DataFrame):
        logger = logging.getLogger("pipeline")
        # in-sample (fitted)
        df_fit = feats[['sku','data']].copy()
        X_all  = feats.drop(columns=['sku','data','venda'])
        df_fit['lgbm_fitted'] = self.lgbm.predict(X_all)
        df_fit['cb_fitted']   = self.cb.predict(X_all)

        # futuro
        last = feats['data'].max()
        future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=self.horizon)
        rows = []
        for d in future_dates:
            row = feats[feats['data']==last].copy()
            row['data'] = d
            rows.append(row)
        fut_feats = pd.concat(rows, ignore_index=True)
        X_fut = fut_feats.drop(columns=['sku','data','venda'])
        fut = pd.DataFrame({
            'sku': fut_feats['sku'],
            'data': fut_feats['data'],
            'lgbm_forecast': self.lgbm.predict(X_fut),
            'cb_forecast':   self.cb.predict(X_fut)
        })

        # concat
        df_out = (
            hist.rename(columns={'venda':'realized'})
                .merge(df_fit, on=['sku','data'], how='left')
                .merge(fut,    on=['sku','data'], how='outer')
        ).sort_values('data').reset_index(drop=True)

        logger.info(f"Gerado forecast SKU={self.sku} com {df_out.shape[0]} linhas")
        return df_out

    def save(self, df: pd.DataFrame):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_csv, index=False)
        logging.getLogger("pipeline").info(f"Forecast SKU salvo em {self.out_csv}")

    def run(self):
        self.load_models()
        hist, feats = self.load_data()
        df_out = self.forecast(hist, feats)
        self.save(df_out)
        logging.getLogger("pipeline").info(f"==== FIM FORECAST SKU={self.sku} ====")


def main():
    parser = argparse.ArgumentParser(description="Forecast por SKU")
    parser.add_argument("sku", type=int, help="SKU para gerar forecast")
    parser.add_argument("--hist",       default="data/processed/vendas_processed.csv")
    parser.add_argument("--features",   default="data/features/vendas_features.csv")
    parser.add_argument("--lgbm-model", default="data/models/lgbm_model.pkl")
    parser.add_argument("--cb-model",   default="data/models/catboost_model.cbm")
    parser.add_argument("--out",        default="data/models/forecast_sku.csv",
                        help="arquivo de saída; será renomeado para forecast_sku_<SKU>.csv")
    parser.add_argument("--log",        default="data/logs/pipeline.log")
    parser.add_argument("--horizon",    type=int, default=7)
    args = parser.parse_args()

    out_file = Path(args.out).parent / f"forecast_sku_{args.sku}.csv"
    forecaster = SKUForecaster(
        sku=args.sku,
        hist_csv=args.hist,
        features_csv=args.features,
        lgbm_model=args.lgbm_model,
        cb_model=args.cb_model,
        out_csv=str(out_file),
        log_path=args.log,
        horizon=args.horizon
    )
    forecaster.run()

if __name__ == "__main__":
    main()
