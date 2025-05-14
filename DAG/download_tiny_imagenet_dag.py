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
    dag_id="download_tiny_imagenet",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["imagenet"]
) as dag:

    download_tiny_imagenet = BashOperator(
        task_id="download_tiny_imagenet",
        bash_command="cd /home/saatarko/PycharmProjects/transfer_testing && dvc repro download_and_prepare_tiny_imagenet_stage",
        doc_md = "**Загрузка и отделение 20 классов у сета Tiny**"
    )


    download_tiny_imagenet