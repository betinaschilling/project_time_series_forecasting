# src/features/make_features.py

import logging
from pathlib import Path
import argparse

from pyspark.sql import SparkSession, Window, functions as F
from delta import configure_spark_with_delta_pip

class FeatureEngineer:
    def __init__(
        self,
        delta_input: str,
        csv_output: str,
        delta_output: str,
        log_path: str,
        lags: list[int],
        windows: list[int]
    ):
        self.delta_input = delta_input
        self.csv_output = csv_output
        self.delta_output = delta_output
        self.log_path = log_path
        self.lags = lags
        self.windows = windows
        self.spark = None
        self._setup_logging()
        self._init_spark()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("pipeline").info("==== CONFIGURAÇÃO DE LOG DE FEATURE ENGINEERING ====")

    def _init_spark(self):
        builder = (
            SparkSession.builder
                .appName("forecast_feature_engineering")
                .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .config("spark.sql.debug.maxToStringFields", "500")
        )
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        logging.getLogger("pipeline").info("SparkSession com suporte a Delta iniciada para features")

    def load_data(self):
        logger = logging.getLogger("pipeline")
        logger.info(f"Features: lendo Delta de entrada em {self.delta_input}")
        df = (
            self.spark.read
                .format("delta")
                .load(self.delta_input)
                .select(
                    "sku",
                    F.col("data").cast("date").alias("data"),
                    F.col("venda").alias("venda"),
                    F.col("is_imputed").alias("is_imputed")
                )
        )
        cnt = df.count()
        logger.info(f"Features: dados brutos carregados ({cnt} linhas)")
        return df

    def add_imputation_flag(self, df):
        logger = logging.getLogger("pipeline")
        if "is_imputed" not in df.columns:
            df = df.withColumn("is_imputed", F.lit(False))
        df = df.withColumn("was_imputed", F.col("is_imputed").cast("int"))
        logger.info("Features: flag de imputação criado")
        return df

    def add_lag_features(self, df):
        logger = logging.getLogger("pipeline")
        w = Window.partitionBy("sku").orderBy("data")
        for lag in self.lags:
            df = df.withColumn(f"lag_{lag}", F.lag("venda", lag).over(w))
        df = df.fillna({f"lag_{lag}": 0 for lag in self.lags})
        logger.info(f"Features: lags {self.lags} criados")
        return df

    def add_rolling_features(self, df):
        logger = logging.getLogger("pipeline")
        w = Window.partitionBy("sku").orderBy("data").rowsBetween(-max(self.windows)-1, -1)
        for win in self.windows:
            df = df.withColumn(
                f"roll_mean_{win}",
                F.avg("venda").over(
                    Window.partitionBy("sku")
                          .orderBy("data")
                          .rowsBetween(-win, -1)
                )
            )
            df = df.fillna({f"roll_mean_{win}": 0})
        logger.info(f"Features: rolling windows {self.windows} criados")
        return df

    def add_calendar_features(self, df):
        logger = logging.getLogger("pipeline")
        df = df.withColumn("weekday", F.dayofweek("data") - 2)  # segunda=0,...domingo=6
        df = df.withColumn("is_weekend", F.when(F.col("weekday").isin([5,6]), 1).otherwise(0))
        df = df.withColumn("month", F.month("data"))
        logger.info("Features: variáveis de calendário criadas")
        return df

    def run(self):
        logger = logging.getLogger("pipeline")
        logger.info("==== INÍCIO DA FEATURE ENGINEERING ====")
        df = self.load_data()
        df = self.add_imputation_flag(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_calendar_features(df)

        # salvar CSV
        Path(self.csv_output).parent.mkdir(parents=True, exist_ok=True)
        df.coalesce(1) \
          .write \
          .mode("overwrite") \
          .option("header", True) \
          .csv(self.csv_output)
        logger.info(f"Features: CSV salvo em {self.csv_output}")

        # salvar Delta
        Path(self.delta_output).parent.mkdir(parents=True, exist_ok=True)
        df.write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .save(self.delta_output)
        logger.info(f"Features: Delta salvo em {self.delta_output}")

        logger.info("==== FIM DA FEATURE ENGINEERING ====")
        self.spark.stop()
        logger.info("SparkSession finalizada")

def main():
    parser = argparse.ArgumentParser(description="Geração de features para forecasting")
    parser.add_argument(
        "--in-delta",
        default="data/processed/vendas_processed.delta",
        help="caminho do Delta de entrada"
    )
    parser.add_argument(
        "--out-csv",
        default="data/features/vendas_features.csv",
        help="onde salvar o CSV de features"
    )
    parser.add_argument(
        "--out-delta",
        default="data/features/vendas_features.delta",
        help="onde salvar o Delta de features"
    )
    parser.add_argument(
        "--log",
        default="data/logs/pipeline.log",
        help="caminho do arquivo de log"
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 7, 14],
        help="lista de lags a gerar"
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[7, 14, 30],
        help="lista de janelas rolling"
    )
    args = parser.parse_args()

    fe = FeatureEngineer(
        delta_input=args.in_delta,
        csv_output=args.out_csv,
        delta_output=args.out_delta,
        log_path=args.log,
        lags=args.lags,
        windows=args.windows
    )
    fe.run()

if __name__ == "__main__":
    main()
