from http import HTTPStatus
from typing import Annotated
import numpy as np
import cv2
import httpx
import base64

from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile

from deep_learning_backend.schemas import (
    Message,
)
from deep_learning_backend.settings import Settings

router = APIRouter(prefix='/dl_prediction', tags=['dl_prediction'])


@router.post(
    '/resnet50',
    response_model=Message,
    status_code=HTTPStatus.CREATED,
)
async def prediction_resnet50(
    file: UploadFile = File(...),
):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imwrite('example_image_resnet50.png', img)

    image_encode = cv2.imencode(".png", np.asarray(img))[1]
    image_str = str(base64.b64encode(image_encode))[2:]

    client = httpx.Client(follow_redirects = True)
    response = client.post(f'{Settings().MODEL_API_URL}/dl_prediction/resnet50', json = {'message': image_str})

    return {'message': str(response.status_code) + ' ' + str(response.json()['message'])}


@router.post(
    '/resnet18',
    response_model=Message,
    status_code=HTTPStatus.CREATED,
)
async def prediction_resnet18(
    file: UploadFile = File(...),
):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imwrite('example_image_resnet18.png', img)

    image_encode = cv2.imencode(".png", np.asarray(img))[1]
    image_str = str(base64.b64encode(image_encode))[2:]

    client = httpx.Client(follow_redirects = True)
    response = client.post(f'{Settings().MODEL_API_URL}/dl_prediction/resnet18', json = {'message': image_str})

    return {'message': str(response.status_code) + ' ' + str(response.json()['message'])}
