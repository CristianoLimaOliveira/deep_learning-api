from datetime import datetime, timedelta, timezone
from random import randint

import factory
import factory.fuzzy

from deep_learning_backend.models import (
    DeepLearningTasks,
    TaskState,
    User,
)


class UserFactory(factory.Factory):
    class Meta:
        model = User

    fullname = factory.Sequence(lambda n: f'test{n}')
    email = factory.LazyAttribute(lambda obj: f'{obj.fullname}@test.com')
    password = factory.LazyAttribute(lambda obj: f'{obj.fullname}@example.com')


class TaskFactory(factory.Factory):
    class Meta:
        model = DeepLearningTasks

    user_id = (1,)
    name = factory.Faker('text')
    description = factory.Faker('text')
    state = TaskState.waiting
    config = {
        'components': [randint(0, 100) for _ in range(5)],
        'initial_timestamp': str(
            datetime.now(timezone.utc) - timedelta(hours=3)
        ),
        'final_timestamp': str(
            datetime.now(timezone.utc) - timedelta(hours=1)
        ),
        'target_component': randint(0, 100),
    }
