from http import HTTPStatus

from deep_learning_backend.models import User


def test_create_user(client, mock_db_time):
    with mock_db_time(model=User):
        response = client.post(
            '/users/',
            json={
                'fullname': 'alice',
                'email': 'alice@example.com',
                'password': 'secret',
            },
        )
        assert response.status_code == HTTPStatus.CREATED
        assert response.json() == {
            'fullname': 'alice',
            'email': 'alice@example.com',
            'id': 1,
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00',
        }


def test_read_users(client, mock_db_time):
    with mock_db_time(model=User):
        response = client.post(
            '/users/',
            json={
                'fullname': 'test0',
                'email': 'test0@test.com',
                'password': 'secret',
            },
        )
        assert response.status_code == HTTPStatus.CREATED
        response = client.get('/users')
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {
            'users': [
                {
                    'created_at': '2024-01-01T00:00:00',
                    'email': 'test0@test.com',
                    'fullname': 'test0',
                    'id': 1,
                    'updated_at': '2024-01-01T00:00:00',
                },
            ]
        }


def test_update_user(client, user, token, mock_db_time):
    with mock_db_time(model=User):
        response = client.put(
            f'/users/{user.id}',
            headers={'Authorization': f'Bearer {token}'},
            json={
                'fullname': 'bob',
                'email': 'bob@example.com',
                'password': 'mynewpassword',
            },
        )
        assert response.status_code == HTTPStatus.OK
        dict_response = response.json()
        del dict_response['updated_at']
        assert dict_response == {
            'fullname': 'bob',
            'email': 'bob@example.com',
            'id': user.id,
            'created_at': '2024-01-01T00:00:00',
        }


def test_update_integrity_error(client, user, other_user, token):
    response_update = client.put(
        f'/users/{user.id}',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'fullname': other_user.fullname,
            'email': 'bob@example.com',
            'password': 'mynewpassword',
        },
    )

    assert response_update.status_code == HTTPStatus.CONFLICT
    assert response_update.json() == {
        'detail': 'Username or Email already exists'
    }


def test_delete_user(client, user, token):
    response = client.delete(
        f'/users/{user.id}',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'message': 'User deleted'}


def test_update_user_with_wrong_user(client, other_user, token):
    response = client.put(
        f'/users/{other_user.id}',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'fullname': 'bob',
            'email': 'bob@example.com',
            'password': 'mynewpassword',
        },
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {'detail': 'Not enough permissions'}


def test_delete_user_wrong_user(client, other_user, token):
    response = client.delete(
        f'/users/{other_user.id}',
        headers={'Authorization': f'Bearer {token}'},
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {'detail': 'Not enough permissions'}
