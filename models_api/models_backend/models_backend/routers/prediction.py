from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
import torchvision
from torch import nn
import torch

from models_backend.schemas import (
    Message,
)
from models_backend.utils import (
    from_base64_to_image,
    get_device,
    get_categories,
    load_model,
    prediction_normalize
)

router = APIRouter(prefix='/dl_prediction', tags=['dl_prediction'])


@router.post(
    '/resnet50',
    response_model=Message,
    status_code=HTTPStatus.CREATED,
)
def prediction_resnet50(
    image_str: Message,
):
    finetune_net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    epoch, model, train_losses, test_losses, optimizer, early_stopping = load_model(finetune_net, './resnet50/bestmodel.pt', list(range(torch.cuda.device_count())))
    model = model.eval().to(get_device())

    image = from_base64_to_image(image_str.message)

    image = prediction_normalize(image).unsqueeze(0)

    with torch.inference_mode():
        result = model(image.to(get_device()))
    
    probs = torch.nn.functional.softmax(result, dim=1).to('cpu')

    return {'message': str(probs) + ' ' + str(torch.argmax(probs[0])) + ' ' + str(get_categories()[torch.argmax(probs[0]).item()])}


@router.post(
    '/resnet18',
    response_model=Message,
    status_code=HTTPStatus.CREATED,
)
def prediction_resnet18(
    image_str: Message,
):
    finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    epoch, model, train_losses, test_losses, optimizer, early_stopping = load_model(finetune_net, './resnet18/bestmodel.pt', list(range(torch.cuda.device_count())))
    model = model.eval().to(get_device())

    image = from_base64_to_image(image_str.message)

    image = prediction_normalize(image).unsqueeze(0)

    with torch.inference_mode():
        result = model(image.to(get_device()))
    
    probs = torch.nn.functional.softmax(result, dim=1).to('cpu')

    return {'message': str(probs) + ' ' + str(torch.argmax(probs[0])) + ' ' + str(get_categories()[torch.argmax(probs[0]).item()])}
