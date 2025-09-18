from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from deep_learning_backend.models import TaskState


class Message(BaseModel):
    message: str


class FilterPage(BaseModel):
    offset: int | None = None
    limit: int | None = None


class FilterTimestamp(FilterPage):
    initial_timestamp: datetime | None = None
    final_timestamp: datetime | None = None


class FileSchema(BaseModel):
    title: str
    configs: dict


class FilePublic(BaseModel):
    id: int
    created_at: datetime
    title: str
    configs: dict


class FileList(BaseModel):
    files: list[FilePublic]


class UserSchema(BaseModel):
    fullname: str
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: int
    fullname: str
    email: EmailStr
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class UserList(BaseModel):
    users: list[UserPublic]


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserPublic


class TokenData(BaseModel):
    username: str | None = None


class DeepLearningTaskSchema(BaseModel):
    name: str
    description: str | None = None
    config: dict


class DeepLearningTaskPublic(DeepLearningTaskSchema):
    id: int
    state: TaskState
    created_at: datetime
    updated_at: datetime


class DeepLearningTaskList(BaseModel):
    tasks: list[DeepLearningTaskPublic]


class DeepLearningTaskUpdate(BaseModel):
    name: str | None = None
    state: TaskState | None = None
    description: str | None = None
    config: dict | None = None


class FilterTask(FilterTimestamp):
    name: str | None = None
    description: str | None = None
    state: TaskState | None = None
