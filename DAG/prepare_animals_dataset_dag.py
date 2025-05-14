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
    dag_id="prepare_animals_dataset",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["animals"]
) as dag:

    prepare_animals_dataset = BashOperator(
        task_id="prepare_animals_dataset",
        bash_command="cd /home/saatarko/PycharmProjects/transfer_testing && dvc repro prepare_animals_dataset_stage",
        doc_md = "**Подготовка датасета animals для обучения**"
    )


    prepare_animals_dataset