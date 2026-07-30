from setuptools import find_packages, setup


setup(
    name="projeto_forecast",
    version="0.2.0",
    description="Pipeline de previsão de demanda e explicabilidade",
    author="Cheila Santos",
    author_email="betina.ssc@gmail.com",
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.23",
        "statsmodels>=0.14",
        "lightgbm>=4.0",
        "catboost>=1.0",
        "scikit-learn>=1.0",
        "joblib>=1.3",
        "streamlit>=1.36",
        "altair>=5.0",
        "pyarrow>=8.0",
        "deltalake>=0.13",
        "pyspark>=3.4",
        "delta-spark>=2.3",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "forecast-load = ingestion.loader:main",
            "forecast-clean = preprocessing.clean:main",
            "forecast-features = features.make_features:main",
            "forecast-train-lgbm = models.train_lgbm:main",
            "forecast-train-cb = models.train_catboost:main",
            "forecast-reconcile = models.reconcile:main",
            "forecast-evaluate = evaluation.metrics:main",
            "forecast-explain = explainability.generate_shap:main",
            "forecast-sku = scripts.run_sku_forecaster:main",
            "forecast-dashboard = visualization.app:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
