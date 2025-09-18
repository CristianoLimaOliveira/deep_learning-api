from http import HTTPStatus
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
import torch
from torch import nn
import torchvision

from models_backend.schemas import (
    Message,
    TrainingSchema
)

from models_backend.utils import (
    get_device,
    EarlyStopping,
    get_train_and_test_dataloader,
    start_training
)

router = APIRouter(prefix='/dl_training', tags=['dl_training'])


@router.post(
    '/resnet50',
    response_model=Message,
    status_code=HTTPStatus.CREATED,
)
def training_resnet50(
    config: TrainingSchema
):
    os.makedirs('./resnet50', exist_ok=True)

    finetune_net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)

    model_res50 = finetune_net.to(get_device())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_res50.parameters(), lr=config.learning_rate)
    num_epochs = config.num_epochs

    early_stopping = EarlyStopping()

    train_loader, test_loader = get_train_and_test_dataloader(batch_size=config.batch_size)
    train_losses, test_losses = start_training(get_device(), num_epochs, model_res50, train_loader, test_loader, optimizer, criterion, early_stopping, path_early_stopping='./resnet50')

    return {'message': 'ok'}


@router.post(
    '/resnet18',
    response_model=Message,
    status_code=HTTPStatus.CREATED,
)
def training_resnet18(
    config: TrainingSchema
):
    os.makedirs('./resnet18', exist_ok=True)

    finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)

    model_res18 = finetune_net.to(get_device())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_res18.parameters(), lr=config.learning_rate)
    num_epochs = config.num_epochs

    early_stopping = EarlyStopping()

    train_loader, test_loader = get_train_and_test_dataloader(batch_size=config.batch_size)
    train_losses, test_losses = start_training(get_device(), num_epochs, model_res18, train_loader, test_loader, optimizer, criterion, early_stopping, path_early_stopping='./resnet18')

    return {'message': 'ok'}
