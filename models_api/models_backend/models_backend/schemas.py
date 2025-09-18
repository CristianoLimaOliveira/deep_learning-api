from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class Message(BaseModel):
    message: str


class TrainingSchema(BaseModel):
    num_epochs: int
    learning_rate: float
    batch_size: int
