# src/evaluation/metrics.py
import logging
from pathlib import Path
import argparse
import sys
import math
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MetricsEvaluator:
    def __init__(self, hist_csv, ml_csv, cb_csv, reconciled_csv, out_csv, log_path):
        self.hist_csv       = hist_csv
        self.ml_csv         = ml_csv
        self.cb_csv         = cb_csv
        self.reconciled_csv = reconciled_csv
        self.out_csv        = Path(out_csv)
        self.log_path       = log_path
        self._setup_logging()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("pipeline").info("==== CONFIGURAÇÃO DE LOG DE AVALIAÇÃO ====")

    def _read_csv(self, path, parse_dates=['data']):
        p = Path(path)
        file = next(p.glob("*.csv")) if p.is_dir() else p
        df = pd.read_csv(file, parse_dates=parse_dates)
        logging.getLogger("pipeline").info(f"Lido {file}: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df

    def load_all(self):
        log = logging.getLogger("pipeline")
        log.info("==== CARREGANDO DADOS PARA AVALIAÇÃO ====")
        hist        = self._read_csv(self.hist_csv)
        ml_forecast = self._read_csv(self.ml_csv)
        cb_forecast = self._read_csv(self.cb_csv)
        rec_forecast= self._read_csv(self.reconciled_csv)


        ml_forecast  = ml_forecast.rename(columns={'ml_pred':'pred'})
        cb_forecast  = cb_forecast.rename(columns={'cb_pred':'pred'})
        rec_forecast = rec_forecast.rename(columns={'reconciled':'pred'})

        return hist, ml_forecast, cb_forecast, rec_forecast

    def _compute_metrics(self, y_true, y_pred):
        mse   = mean_squared_error(y_true, y_pred)
        rmse  = math.sqrt(mse)
        mae   = mean_absolute_error(y_true, y_pred)
        r2    = r2_score(y_true, y_pred)
        mape  = (abs((y_true - y_pred)/y_true.replace(0,1)).mean())*100
        wmape = abs(y_true - y_pred).sum() / y_true.sum() * 100
        return {'rmse':rmse, 'mae':mae, 'r2':r2, 'mape':mape, 'wmape':wmape}

    def compute(self):
        log = logging.getLogger("pipeline")
        hist, ml, cb, rec = self.load_all()
        records = []
        for name, df_pred in [('lgbm',ml),('catboost',cb),('reconciled',rec)]:
            df_g = hist.merge(df_pred, on=['sku','data'], how='inner')
            mg = self._compute_metrics(df_g['venda'], df_g['pred'])
            log.info(f"{name} granular: {mg}")
            records.append({'model':name,'level':'granular',**mg})

            agg_r = hist.groupby('data')['venda'].sum().reset_index()
            agg_p = df_pred.groupby('data')['pred'].sum().reset_index()
            df_a = agg_r.merge(agg_p, on='data')
            ma = self._compute_metrics(df_a['venda'], df_a['pred'])
            log.info(f"{name} aggregate: {ma}")
            records.append({'model':name,'level':'aggregate',**ma})

        return pd.DataFrame.from_records(records, columns=['model','level','rmse','mae','mape','wmape','r2'])

    def save(self, df_metrics):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(self.out_csv, index=False)
        logging.getLogger("pipeline").info(f"Metrics report salvo em {self.out_csv}")

    def run(self):
        log = logging.getLogger("pipeline")
        log.info("==== INÍCIO DA AVALIAÇÃO DE MÉTRICAS ====")
        dfm = self.compute()
        self.save(dfm)
        log.info("==== FIM DA AVALIAÇÃO DE MÉTRICAS ====")

def main():
    parser = argparse.ArgumentParser(description="Avaliação de métricas de forecasting")
    parser.add_argument("--hist",       default="data/processed/vendas_processed.csv")
    parser.add_argument("--ml",         default="data/models/ml_forecast.csv")
    parser.add_argument("--cb",         default="data/models/catboost_forecast.csv")
    parser.add_argument("--reconciled", default="data/models/reconciled_sku_forecast.csv")
    parser.add_argument("--out",        default="data/models/metrics_report.csv")
    parser.add_argument("--log",        default="data/logs/pipeline.log")
    args = parser.parse_args()

    evaluator = MetricsEvaluator(
        hist_csv=args.hist,
        ml_csv=args.ml,
        cb_csv=args.cb,
        reconciled_csv=args.reconciled,
        out_csv=args.out,
        log_path=args.log
    )
    evaluator.run()

if __name__ == "__main__":
    main()
