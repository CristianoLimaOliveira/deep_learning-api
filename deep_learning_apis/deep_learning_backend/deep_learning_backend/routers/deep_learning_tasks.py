from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, select
from sqlalchemy.orm import Session

from deep_learning_backend.database_session import get_session
from deep_learning_backend.models import (
    DeepLearningTasks,
    TaskState,
    User,
)
from deep_learning_backend.schemas import (
    DeepLearningTaskList,
    DeepLearningTaskPublic,
    DeepLearningTaskSchema,
    DeepLearningTaskUpdate,
    FilterTask,
    Message,
)
from deep_learning_backend.security import get_current_user

router = APIRouter(prefix='/dl_tasks', tags=['dl_tasks'])
SessionType = Annotated[Session, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post(
    '/',
    response_model=DeepLearningTaskPublic,
    status_code=HTTPStatus.CREATED,
)
def create_dl_task(
    task: DeepLearningTaskSchema,
    user: CurrentUser,
    session: SessionType,
):
    db_task = session.scalar(
        select(DeepLearningTasks).where(
            (DeepLearningTasks.user_id == user.id)
            & (DeepLearningTasks.name == task.name)
        )
    )

    if db_task:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT, detail='Task name already exists.'
        )

    db_task = DeepLearningTasks(
        user_id=user.id,
        name=task.name,
        description=task.description,
        config=task.config,
        state=TaskState.waiting,
    )

    session.add(db_task)
    session.commit()
    session.refresh(db_task)

    return db_task


@router.patch('/{task_id}', response_model=Message)
def update_task(
    task_id: int,
    task_update: DeepLearningTaskUpdate,
    session: SessionType,
    user: CurrentUser,
):
    task = session.scalar(
        select(DeepLearningTasks).where(DeepLearningTasks.id == task_id)
    )
    if not task:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail='task not found.',
        )

    if task_update.name is not None:
        task.name = task_update.name

    if task_update.state is not None:
        task.state = task_update.state

    if task_update.description is not None:
        task.description = task_update.description

    if task_update.config is not None:
        task.config = task_update.config

    session.commit()
    session.refresh(task)

    return {'message': 'Task updated successfully.'}


@router.get(
    '/', response_model=DeepLearningTaskList, status_code=HTTPStatus.OK
)
def list_tasks(
    session: SessionType,
    user: CurrentUser,
    task_filter: Annotated[FilterTask, Query()],
):
    query = select(DeepLearningTasks).where(
        DeepLearningTasks.user_id == user.id
    )

    if task_filter.name:
        query = query.filter(DeepLearningTasks.name.contains(task_filter.name))

    if task_filter.description:
        query = query.filter(
            DeepLearningTasks.description.contains(task_filter.description)
        )

    if task_filter.state:
        query = query.filter(DeepLearningTasks.state == task_filter.state)

    if task_filter.initial_timestamp:
        query = query.filter(
            (DeepLearningTasks.created_at >= task_filter.initial_timestamp)
        )

    if task_filter.final_timestamp:
        query = query.filter(
            (DeepLearningTasks.created_at <= task_filter.final_timestamp)
        )
    if task_filter.offset:
        query = query.offset(task_filter.offset)
    if task_filter.limit:
        query = query.limit(task_filter.limit)

    tasks = session.scalars(query).all()

    return {'tasks': tasks}


@router.get(
    '/next_task',
    response_model=DeepLearningTaskPublic,
    status_code=HTTPStatus.OK,
)
def next_waiting_task(
    session: SessionType,
    user: CurrentUser,
):
    db_task = session.scalars(
        select(DeepLearningTasks)
        .where(
            (DeepLearningTasks.user_id == user.id)
            & (DeepLearningTasks.state == TaskState.waiting)
        )
        .order_by(asc(DeepLearningTasks.created_at))
    ).first()

    if not db_task:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail='No waiting tasks were found.',
        )

    return db_task


@router.delete('/{task_id}', response_model=Message, status_code=HTTPStatus.OK)
def delete_task(task_id: int, session: SessionType, user: CurrentUser):
    task = session.scalar(
        select(DeepLearningTasks).where(DeepLearningTasks.id == task_id)
    )

    if not task:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail='Task not found.'
        )

    if task.user.id != user.id:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED, detail='Task unauthorized.'
        )

    session.delete(task)
    session.commit()

    return {'message': 'Task has been deleted successfully.'}
