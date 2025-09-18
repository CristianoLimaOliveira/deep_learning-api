from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import json
from api_client import TaskState, get_highest_priority, update_to_doing, process_record, update_to_done
import httpx

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=50),
}



with DAG(
    'record_processing_dag',
    default_args = default_args,
    description = 'Process record with state management',
    schedule_interval = timedelta(minutes = 5),
    catchup = False,
    render_template_as_native_obj=True,
) as dag:

    # Task to update state to DOING
    start_processing = ShortCircuitOperator(
        task_id='next_task',
        python_callable=get_highest_priority,
    )

    # Main processing task
    update_state_to_doing = PythonOperator(
        task_id='update_to_doing',
        python_callable=update_to_doing,
    )

    # Main processing task
    process_task = PythonOperator(
        task_id='process_record',
        python_callable=process_record,
    )

    # Task to update state to DONE
    finish_processing = PythonOperator(
        task_id='update_to_done',
        python_callable=update_to_done,
    )

    # Set up task dependencies
    start_processing >> update_state_to_doing >> process_task >> finish_processing