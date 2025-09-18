from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from deep_learning_backend.settings import Settings

engine = create_engine(Settings().DATABASE_URL)
engine_test = create_engine(Settings().DATABASE_URL_TEST)


def get_session():
    with Session(engine) as session:
        yield session
