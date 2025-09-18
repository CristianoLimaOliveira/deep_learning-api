from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from deep_learning_backend.database_session import get_session
from deep_learning_backend.models import User
from deep_learning_backend.schemas import (
    FilterTimestamp,
    Message,
    UserList,
    UserPublic,
    UserSchema,
)
from deep_learning_backend.security import (
    get_current_user,
    get_password_hash,
)

router = APIRouter(prefix='/users', tags=['users'])
SessionType = Annotated[Session, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post('/', status_code=HTTPStatus.CREATED, response_model=UserPublic)
def create_user(user: UserSchema, session: SessionType):
    db_user = session.scalar(
        select(User).where(
            (User.fullname == user.fullname) | (User.email == user.email)
        )
    )

    if db_user:
        if db_user.fullname == user.fullname:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail='Username already exists.',
            )
        elif db_user.email == user.email:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail='Email already exists',
            )

    hashed_password = get_password_hash(user.password)

    db_user = User(
        email=user.email,
        fullname=user.fullname,
        password=hashed_password,
    )

    session.add(db_user)
    session.commit()
    session.refresh(db_user)

    return db_user


@router.get('/', response_model=UserList)
def read_users(
    session: SessionType, filter_users: Annotated[FilterTimestamp, Query()]
):
    query = select(User)
    if filter_users.initial_timestamp:
        query = query.filter(
            (User.created_at >= filter_users.initial_timestamp)
        )
    if filter_users.final_timestamp:
        query = query.filter((User.created_at <= filter_users.final_timestamp))
    if filter_users.offset:
        query = query.offset(filter_users.offset)
    if filter_users.limit:
        query = query.limit(filter_users.limit)
    users = session.scalars(query).all()

    return {'users': users}


@router.put('/{user_id}', response_model=UserPublic)
def update_user(
    user_id: int,
    user: UserSchema,
    session: SessionType,
    current_user: CurrentUser,
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN, detail='Not enough permissions'
        )

    try:
        current_user.fullname = user.fullname
        current_user.password = get_password_hash(user.password)
        current_user.email = user.email
        session.commit()
        session.refresh(current_user)

        return current_user

    except IntegrityError:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail='Username or Email already exists',
        )


@router.delete('/{user_id}', response_model=Message)
def delete_user(
    user_id: int,
    session: SessionType,
    current_user: CurrentUser,
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN, detail='Not enough permissions'
        )

    session.delete(current_user)
    session.commit()

    return {'message': 'User deleted'}
