from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deep_learning_backend.routers import auth, deep_learning_tasks, users, prediction
from deep_learning_backend.schemas import Message

app = FastAPI(
    title='Deep Learning - API',
    description='Descrição da API',
    version='0.0.1',
    contact={
        'name': 'Cristiano Lima Oliveira',
        'email': 'clo@cin.ufpe.br',
    },
    license_info={
        'name': 'Apache 2.0',
        'url': 'https://www.apache.org/licenses/LICENSE-2.0.html',
    },
)


allow_all = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_all,
    allow_credentials=True,
    allow_methods=allow_all,
    allow_headers=allow_all,
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(deep_learning_tasks.router)
app.include_router(prediction.router)


@app.get('/', status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {'message': 'Bem-vindo à Deep Learning API! :)'}
