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
    dag_id="train_nn",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["mnist"]
) as dag:

    train_nn = BashOperator(
        task_id="train_nn",
        bash_command="cd /home/saatarko/PycharmProjects/transfer_testing && dvc repro train_nn_stage",
        doc_md = "**Обучение модели на mnist**"
    )


    train_nn