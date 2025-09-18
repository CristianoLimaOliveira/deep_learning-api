import httpx
from typing import Dict, Any
from enum import Enum
import logging
from airflow.models import Variable

class TaskState(str, Enum):
    waiting = "waiting"
    doing = "doing"
    done = "done"
    fail = "fail"

def auth_user():
    client = httpx.Client(follow_redirects = True)
    access_token = client.post(f'{Variable.get('main_backend_url')}/auth/token', data = {'username': Variable.get('auth_username'), 'password': Variable.get('auth_password')}).json().get('access_token')
    headers = {
        'Authorization':f'Bearer {access_token}',
        'Content-Type':'application/json'
    }
    return client, headers
    
def get_highest_priority(**kwargs):
    client, headers = auth_user()
    response = client.get(f'{Variable.get('main_backend_url')}/dl_tasks/next_task', headers = headers)
    kwargs['ti'].xcom_push(key="next_task", value=response.json())
    return response.status_code != 404

def update_to_doing(**kwargs):
    client, headers = auth_user()
    task = kwargs['ti'].xcom_pull(key="next_task", task_ids="next_task")
    task.update({'state': TaskState.doing})
    client.patch(f'{Variable.get('main_backend_url')}/dl_tasks/' + str(task.get('id', '')), json = task, headers = headers)

def update_to_done(**kwargs):
    client, headers = auth_user()
    task = kwargs['ti'].xcom_pull(key="next_task", task_ids="next_task")
    task.update({'state': TaskState.done})
    client.patch(f'{Variable.get('main_backend_url')}/dl_tasks/' + str(task.get('id', '')), json = task, headers = headers)

def process_record(**kwargs):
    """
    Main processing logic for the record
    Replace this with your actual processing logic
    """
    client, headers = auth_user()
    task = kwargs['ti'].xcom_pull(key="next_task", task_ids="next_task")
    try:
        if task['config']['model'].lower() == 'resnet50':
            model = 'resnet50'
        elif task['config']['model'].lower() == 'resnet18':
            model = 'resnet18'
        else:
            task.update({'state': TaskState.fail})
            client.patch(f'{Variable.get('main_backend_url')}/dl_tasks/' + str(task.get('id', '')), json = task, headers = headers)
            kwargs['ti'].xcom_push(key="process", value={"success": False})

        response = client.post(
                f'http://172.16.1.14:8100/dl_training/{model}',
                json = {
                    'num_epochs': int(task['config']['num_epochs']),
                    'learning_rate': float(task['config']['learning_rate']),
                    'batch_size': int(task['config']['batch_size'])
                }
            )

        kwargs['ti'].xcom_push(key="process", value={"success": True, "response_status_code": response.status_code, "response": response.json()})
    except Exception as e:
        task.update({'state': TaskState.fail})
        client.patch(f'{Variable.get('main_backend_url')}/dl_tasks/' + str(task.get('id', '')), json = task, headers = headers)

        kwargs['ti'].xcom_push(key="process", value={"success": False})
