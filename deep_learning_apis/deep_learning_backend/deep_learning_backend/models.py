from datetime import datetime
from enum import Enum
from typing import List

from sqlalchemy import JSON, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, registry, relationship

table_registry = registry()


class TaskState(str, Enum):
    waiting = 'waiting'
    doing = 'doing'
    done = 'done'
    fail = 'fail'


@table_registry.mapped_as_dataclass
class User:
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(
        init=False, primary_key=True, autoincrement=True, nullable=False
    )
    fullname: Mapped[str] = mapped_column(nullable=False, unique=True)
    email: Mapped[str] = mapped_column(nullable=False, unique=True)
    password: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        init=False,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    deep_learning_tasks: Mapped[List['DeepLearningTasks']] = relationship(
        init=False, back_populates='user', cascade='all, delete-orphan'
    )


@table_registry.mapped_as_dataclass
class DeepLearningTasks:
    __tablename__ = 'deep_learning_tasks'

    id: Mapped[int] = mapped_column(
        init=False, primary_key=True, autoincrement=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', onupdate='cascade', ondelete='cascade'),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=True)
    state: Mapped[TaskState] = mapped_column(nullable=False)
    config: Mapped[JSON] = mapped_column(type_=JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        init=False,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship(
        init=False, back_populates='deep_learning_tasks'
    )

    # Unique constraint
    __table_args__ = (
        UniqueConstraint(
            'user_id',
            'name',
            name='deep_learning_tasks_user_id_key_name_key',
        ),
    )
