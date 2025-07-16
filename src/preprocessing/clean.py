# src/preprocessing/clean.py

import logging
from pathlib import Path
import argparse

from pyspark.sql import SparkSession, functions as F
from delta import configure_spark_with_delta_pip

class DataCleaner:
    def __init__(
        self,
        delta_input: str,
        csv_output: str,
        delta_output: str,
        log_path: str
    ):
        self.delta_input = delta_input
        self.csv_output = csv_output
        self.delta_output = delta_output
        self.log_path = log_path
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
        logging.getLogger("pipeline").info("==== CONFIGURAÇÃO DE LOG DE PREPROCESSAMENTO ====")

    def _init_spark(self):
        builder = (
            SparkSession.builder
                .appName("forecast_data_cleaner")
                .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .config("spark.sql.debug.maxToStringFields", "500")
        )
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        logging.getLogger("pipeline").info("SparkSession com suporte a Delta iniciada")

    def load_data(self):
        logger = logging.getLogger("pipeline")
        logger.info(f"lendo Delta de entrada em {self.delta_input}")
        df = (
            self.spark.read
                .format("delta")
                .load(self.delta_input)
                .select("sku", 
                        F.col("data_venda").cast("date").alias("data"), 
                        "venda"
                )
        )
        cnt = df.count()
        logger.info(f"dados brutos carregados: {cnt} linhas")
        return df

    def clean_data(self, df):
        logger = logging.getLogger("pipeline")

        # agrupa duplicatas somando vendas
        df = df.groupBy("sku", "data").agg(F.sum("venda").alias("venda"))
        logger.info("agrupadas duplicatas por sku+data")

        # calcula intervalos por SKU
        ranges = (
            df.groupBy("sku")
              .agg(
                  F.min("data").alias("start"),
                  F.max("data").alias("end")
              )
              .withColumn("date_range", F.expr("sequence(start, end, interval 1 day)"))
              .select("sku", F.explode("date_range").alias("data"))
        )
        logger.info("gerada malha completa de datas por SKU")

        # faz join e marca imputados
        df_full = ranges.join(df, ["sku", "data"], "left")
        df_full = df_full.withColumn(
            "is_imputed",
            F.when(F.col("venda").isNull(), F.lit(True)).otherwise(F.lit(False))
        )
        logger.info("marcado campo is_imputed")

        # preenche nulos com zero
        df_full = df_full.fillna({"venda": 0})
        logger.info("preenchidos valores nulos de venda com zero")

        # ordena
        df_full = df_full.orderBy("sku", "data")
        logger.info("ordenado por sku e data")

        return df_full

    def run(self):
        logger = logging.getLogger("pipeline")
        logger.info("==== INÍCIO DA ETAPA DE PREPROCESSAMENTO ====")

        df = self.load_data()
        df_clean = self.clean_data(df)

        # salva CSV
        Path(self.csv_output).parent.mkdir(parents=True, exist_ok=True)
        df_clean.coalesce(1) \
                .write \
                .mode("overwrite") \
                .option("header", True) \
                .csv(self.csv_output)
        logger.info(f"CSV processado salvo em {self.csv_output}")

        # salva Delta
        Path(self.delta_output).parent.mkdir(parents=True, exist_ok=True)
        df_clean.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(self.delta_output)
        logger.info(f"Delta processado salvo em {self.delta_output}")

        logger.info("==== FIM DA ETAPA DE PREPROCESSAMENTO ====")
        self.spark.stop()
        logger.info("SparkSession finalizada")

def main():
    parser = argparse.ArgumentParser(description="Pré-processamento Spark de vendas")
    parser.add_argument(
        "--in-delta",
        default="data/interim/vendas_interim.delta",
        help="caminho do Delta interim de entrada"
    )
    parser.add_argument(
        "--out-csv",
        default="data/processed/vendas_processed.csv",
        help="onde salvar o CSV processado"
    )
    parser.add_argument(
        "--out-delta",
        default="data/processed/vendas_processed.delta",
        help="onde salvar o Delta processado"
    )
    parser.add_argument(
        "--log",
        default="data/logs/pipeline.log",
        help="caminho do arquivo de log"
    )
    args = parser.parse_args()

    cleaner = DataCleaner(
        delta_input=args.in_delta,
        csv_output=args.out_csv,
        delta_output=args.out_delta,
        log_path=args.log
    )
    cleaner.run()

if __name__ == "__main__":
    main()
