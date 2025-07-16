# src/ingestion/loader.py

import logging
from pathlib import Path
import argparse
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

class DataLoader:
    def __init__(self, raw_path: str, csv_path: str, delta_path: str, log_path: str):
        self.raw_path = raw_path
        self.csv_path = csv_path
        self.delta_path = delta_path
        self.log_path = log_path
        self.spark = None
        self._setup_logging()

    def _setup_logging(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("configuração de logging concluída")

    def _init_spark(self):
        builder = (
            SparkSession.builder
            .appName("forecast_data_loader")
            .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.debug.maxToStringFields", "500")   # aumenta o limite de campos
        )
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logging.info("SparkSession com suporte a Delta iniciada")

    def load_raw(self):
        df = (
            self.spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(self.raw_path)
        )
        count = df.count()
        cols  = len(df.columns)
        logging.info(f"raw data carregada: {count} linhas, {cols} colunas")
        return df

    def save_csv(self, df):
        out_dir = Path(self.csv_path)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        df.coalesce(1) \
          .write \
          .mode("overwrite") \
          .option("header", True) \
          .csv(str(out_dir))
        logging.info(f"interim CSV salvo em {self.csv_path}")

    def save_delta(self, df):
        out_dir = Path(self.delta_path)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        df.write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .save(str(out_dir))
        logging.info(f"interim Delta salvo em {self.delta_path}")

    def run(self):
        logging.info("==== INÍCIO DA ETAPA DE INGESTÃO ====")
        try:
            self._init_spark()
            df = self.load_raw()
            self.save_csv(df)
            self.save_delta(df)
            logging.info("==== FIM DA ETAPA DE INGESTÃO ====")
            logging.info("ingestão concluída com sucesso")
        except Exception:
            logging.exception("falha na ingestão de dados")
            raise
        finally:
            if self.spark:
                self.spark.stop()
                logging.info("SparkSession finalizada")


def main():
    parser = argparse.ArgumentParser(description="Loader de dados de vendas")
    parser.add_argument(
        "--raw",
        default="data/raw/vendas.csv",
        help="caminho do CSV de entrada"
    )
    parser.add_argument(
        "--out-csv",
        default="data/interim/vendas_interim.csv",
        help="onde salvar o CSV interim (gera pasta com arquivos .csv)"
    )
    parser.add_argument(
        "--out-delta",
        default="data/interim/vendas_interim.delta",
        help="onde salvar o Delta interim (diretório Delta Lake)"
    )
    parser.add_argument(
        "--log",
        default="data/logs/pipeline.log",
        help="caminho do arquivo de log"
    )
    args = parser.parse_args()

    loader = DataLoader(
        raw_path=args.raw,
        csv_path=args.out_csv,
        delta_path=args.out_delta,
        log_path=args.log
    )
    loader.run()


if __name__ == "__main__":
    main()
