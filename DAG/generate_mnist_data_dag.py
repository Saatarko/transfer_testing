from datetime import datetime
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === DAG конфигурация ===
with DAG(
    dag_id="generate_mnist_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["mnist"]
) as dag:

    generate_mnist_data = BashOperator(
        task_id="generate_mnist_data",
        bash_command="cd /home/saatarko/PycharmProjects/transfer_testing && dvc repro generate_mnist_data_stage",
        doc_md = "**Генерация данных mnist**"
    )


    generate_mnist_data