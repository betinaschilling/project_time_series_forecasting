#!/usr/bin/env python
# src/models/reconcile.py

import logging
from pathlib import Path
import argparse

import pandas as pd

class Reconciler:
    def __init__(
        self,
        ml_csv: str,
        cb_csv: str,
        hist_csv: str,
        out_dir: str,
        log_path: str
    ):
        self.ml_csv   = ml_csv
        self.cb_csv   = cb_csv
        self.hist_csv = hist_csv
        self.out_dir  = Path(out_dir)
        self.log_path = log_path
        self._setup_logging()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("pipeline").info("==== CONFIGURAÇÃO DE LOG DE RECONCILIAÇÃO ====")

    def load_data(self):
        logger = logging.getLogger("pipeline")
        logger.info(f"Lendo forecast LightGBM de {self.ml_csv}")
        ml = pd.read_csv(self.ml_csv, parse_dates=['data'])
        logger.info(f"ML forecast: {ml.shape[0]} linhas")

        logger.info(f"Lendo forecast CatBoost de {self.cb_csv}")
        cb = pd.read_csv(self.cb_csv, parse_dates=['data'])
        logger.info(f"CatBoost forecast: {cb.shape[0]} linhas")

        logger.info(f"Lendo histórico processado de {self.hist_csv}")
        p_hist = Path(self.hist_csv)
        if p_hist.is_dir():
            hist_file = next(p_hist.glob("*.csv"))
        else:
            hist_file = p_hist
        hist = pd.read_csv(hist_file, parse_dates=['data'])
        logger.info(f"Histórico: {hist.shape[0]} linhas (arquivo: {hist_file})")
        return ml, cb, hist

    def reconcile(self, ml: pd.DataFrame, cb: pd.DataFrame, hist: pd.DataFrame):
        logger = logging.getLogger("pipeline")

        # SKU-level merge
        df = (
            ml.rename(columns={'ml_pred': 'lgbm_pred'})
              .merge(
                  cb.rename(columns={'cb_pred': 'cb_pred'}),
                  on=['sku', 'data'], how='outer'
              )
        )
        df['lgbm_pred'] = df['lgbm_pred'].fillna(0)
        df['cb_pred']   = df['cb_pred'].fillna(0)

        # reconciled as average of the two methods
        df['reconciled'] = (df['lgbm_pred'] + df['cb_pred']) / 2
        logger.info("Criado campo 'reconciled' = média de lgbm_pred e cb_pred")

        # attach realized values
        real = hist.groupby(['sku', 'data'])['venda'] \
                   .sum().reset_index().rename(columns={'venda': 'realized'})
        df_sku = df.merge(real, on=['sku', 'data'], how='left')
        df_sku['realized'] = df_sku['realized'].fillna(0)

        # daily aggregate summary
        df_daily = (
            df_sku.groupby('data')
                  .agg(
                      realized_total=('realized', 'sum'),
                      lgbm_total=('lgbm_pred', 'sum'),
                      catboost_total=('cb_pred', 'sum'),
                      reconciled_total=('reconciled', 'sum')
                  )
                  .reset_index()
        )
        logger.info("Computado resumo diário agregado")

        return df_sku, df_daily

    def save(self, df_sku: pd.DataFrame, df_daily: pd.DataFrame):
        # ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

        sku_path = self.out_dir / 'reconciled_sku_forecast.csv'
        df_sku.to_csv(sku_path, index=False)
        logging.getLogger("pipeline").info(f"Reconciled SKU-level salvo em {sku_path}")

        daily_path = self.out_dir / 'reconciled_daily_summary.csv'
        df_daily.to_csv(daily_path, index=False)
        logging.getLogger("pipeline").info(f"Reconciled daily summary salvo em {daily_path}")

    def run(self):
        logger = logging.getLogger("pipeline")
        logger.info("==== INÍCIO DA RECONCILIAÇÃO ====")
        ml, cb, hist = self.load_data()
        df_sku, df_daily = self.reconcile(ml, cb, hist)
        self.save(df_sku, df_daily)
        logger.info("==== FIM DA RECONCILIAÇÃO ====")

def main():
    parser = argparse.ArgumentParser(description="Reconciliação de forecasts")
    parser.add_argument(
        "--ml-csv",
        default="data/models/ml_forecast.csv",
        help="forecast granular LightGBM (csv ou pasta contendo part-*.csv)"
    )
    parser.add_argument(
        "--cb-csv",
        default="data/models/catboost_forecast.csv",
        help="forecast granular CatBoost"
    )
    parser.add_argument(
        "--hist",
        default="data/processed/vendas_processed.csv",
        help="histórico processado"
    )
    parser.add_argument(
        "--out-dir",
        default="data/models",
        help="diretório para salvar outputs reconciliados"
    )
    parser.add_argument(
        "--log",
        default="data/logs/pipeline.log",
        help="caminho do arquivo de log"
    )
    args = parser.parse_args()

    reconciler = Reconciler(
        ml_csv=args.ml_csv,
        cb_csv=args.cb_csv,
        hist_csv=args.hist,
        out_dir=args.out_dir,
        log_path=args.log
    )
    reconciler.run()

if __name__ == "__main__":
    main()
