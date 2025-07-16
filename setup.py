from setuptools import setup, find_packages

setup(
    name='projeto_forecast',
    version='0.1',
    description='Pipeline de previsão de demanda modular em Python',
    author='Cheila Santos',
    author_email='betina.ssc@gmail.com',
    python_requires='>=3.8',
    install_requires=[
        'pandas>=2.0',
        'numpy>=1.23',
        'statsmodels>=0.14',
        'lightgbm>=4.0',
        'catboost>=1.0',
        'scikit-learn>=1.0',
        'pyarrow>=8.0.0', 
        'deltalake>=0.13',
        'pyspark>=3.4.0',
        'delta-spark>=2.3.0'
 
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
        entry_points={
        'console_scripts': [
            'forecast-load = ingestion.loader:main',
            'forecast-clean = preprocessing.clean:main',
            'forecast-features = features.make_features:main',
            'forecast-train-lgbm = models.train_lgbm:main',
            'forecast-train-cb = models.train_catboost:main',
            'forecast-reconcile = models.reconcile:main',
            'forecast-evaluate = evaluation.metrics:main',
            'forecast-sku = scripts.run_sku_forecaster:main',
            'forecast-dashboard = visualization.app:main'
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    zip_safe=False,
)