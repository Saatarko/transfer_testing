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
    dag_id="evaluate_no_frease_mnist",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["mnist"]
) as dag:

    evaluate_no_frease_mnist = BashOperator(
        task_id="evaluate_no_frease_mnist",
        bash_command="cd /home/saatarko/PycharmProjects/transfer_testing && dvc repro evaluate_fash_no_frease_model_on_test_stage",
        doc_md = "**Предикты на обученной модели mnist**"
    )


    evaluate_no_frease_mnist