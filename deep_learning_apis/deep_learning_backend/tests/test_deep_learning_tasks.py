import time
from http import HTTPStatus

from deep_learning_backend.models import (
    DeepLearningTasks,
    TaskState,
)
from tests.factories import TaskFactory


def test_create_dl_task(client, token, mock_db_time):
    with mock_db_time(model=DeepLearningTasks):
        response = client.post(
            '/dl_tasks/',
            headers={'Authorization': f'Bearer {token}'},
            json={
                'name': 'Task example',
                'description': 'This is an example of creating one task',
                'config': {
                    'components': [1, 2, 3, 4],
                    'initial_timestamp': '2023-12-04 18:18:04',
                    'final_timestamp': '2024-12-04 18:18:04',
                    'target_component': 3,
                },
            },
        )

        assert response.status_code == HTTPStatus.CREATED
        assert response.json() == {
            'id': 1,
            'name': 'Task example',
            'description': 'This is an example of creating one task',
            'state': 'waiting',
            'config': {
                'components': [1, 2, 3, 4],
                'initial_timestamp': '2023-12-04 18:18:04',
                'final_timestamp': '2024-12-04 18:18:04',
                'target_component': 3,
            },
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00',
        }


def test_list_tasks_should_return_5_tasks(session, client, token, user):
    expected_tasks = 5
    session.bulk_save_objects(
        TaskFactory.create_batch(
            5,
            user_id=user.id,
        )
    )
    session.commit()

    response = client.get(
        '/dl_tasks/',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert len(response.json()['tasks']) == expected_tasks


def test_list_tasks_filter_description_should_return_5_tasks(
    session, client, token, user
):
    expected_tasks = 5
    session.bulk_save_objects(
        TaskFactory.create_batch(
            5,
            user_id=user.id,
            description='description',
        )
    )

    response = client.get(
        '/dl_tasks/?description=desc',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert len(response.json()['tasks']) == expected_tasks


def test_list_tasks_filter_initial_timestamp_should_return_5_tasks(
    session, client, token, user
):
    expected_tasks = 5
    session.bulk_save_objects(
        TaskFactory.create_batch(
            5,
            user_id=user.id,
        )
    )

    response = client.get(
        '/dl_tasks/?initial_timestamp=2023-12-05T01:12:02',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert len(response.json()['tasks']) == expected_tasks


def test_list_tasks_filter_final_timestamp_should_return_5_tasks(
    session, client, token, user
):
    expected_tasks = 5
    session.bulk_save_objects(
        TaskFactory.create_batch(
            5,
            user_id=user.id,
        )
    )

    response = client.get(
        '/dl_tasks/?final_timestamp=2025-12-05T01:12:02',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert len(response.json()['tasks']) == expected_tasks


def test_next_waiting_task(session, client, token, user, mock_db_time):
    with mock_db_time(model=DeepLearningTasks):
        task1 = TaskFactory(
            user_id=user.id,
            name='task 1',
            description='description task 1',
            state='waiting',
            config={
                'components': [1, 2, 3, 4],
                'initial_timestamp': '2023-12-04 18:18:04',
                'final_timestamp': '2024-12-04 18:18:04',
                'target_component': 3,
            },
        )
        session.add(task1)
        session.commit()

        time.sleep(1)

        task2 = TaskFactory(
            user_id=user.id,
            name='task 2',
            description='description task 2',
            state='waiting',
            config={
                'components': [2, 3, 4, 5],
                'initial_timestamp': '2023-10-04 08:08:54',
                'final_timestamp': '2024-10-04 08:08:54',
                'target_component': 2,
            },
        )
        session.add(task2)
        session.commit()

        response = client.get(
            '/dl_tasks/next_task', headers={'Authorization': f'Bearer {token}'}
        )

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {
            'id': 1,
            'name': 'task 1',
            'description': 'description task 1',
            'state': 'waiting',
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00',
            'config': {
                'components': [1, 2, 3, 4],
                'final_timestamp': '2024-12-04 18:18:04',
                'initial_timestamp': '2023-12-04 18:18:04',
                'target_component': 3,
            },
        }


def test_list_tasks_filter_state_should_return_5_tasks(
    session, client, token, user
):
    expected_tasks = 5
    session.bulk_save_objects(
        TaskFactory.create_batch(
            5,
            user_id=user.id,
            state=TaskState.waiting,
        )
    )

    response = client.get(
        '/dl_tasks/?state=waiting',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert len(response.json()['tasks']) == expected_tasks


def test_delete_task(session, client, token, user):
    task = TaskFactory(
        user_id=user.id,
    )

    session.add(task)
    session.commit()

    response = client.delete(
        f'/dl_tasks/{task.id}', headers={'Authorization': f'Bearer {token}'}
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        'message': 'Task has been deleted successfully.'
    }


def test_delete_task_error_taskid(session, client, token, user):
    task = TaskFactory(
        user_id=user.id,
    )

    session.add(task)
    session.commit()

    nonexistent_id = task.id + 1
    response = client.delete(
        f'/dl_tasks/{nonexistent_id}',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {'detail': 'Task not found.'}


def test_delete_task_error_token(session, client, token, user, other_user):
    task = TaskFactory(
        user_id=user.id,
    )

    session.add(task)
    session.commit()

    response = client.post(
        '/auth/token',
        data={
            'username': other_user.email,
            'password': other_user.clean_password,
        },
    )
    other_user_token = response.json()['access_token']

    response = client.delete(
        f'/dl_tasks/{task.id}',
        headers={'Authorization': f'Bearer {other_user_token}'},
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json() == {'detail': 'Task unauthorized.'}
