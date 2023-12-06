import uvicorn
from fastapi import FastAPI, status, Response
from models import Models
from pydantic import BaseModel

app = FastAPI()

models = Models()

class ModelItem(BaseModel):
    model_name: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_name": "LogisticRegression"
                }
            ]
        }
    }

class ModelUpd(BaseModel):
    model_name: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                "model_name": 
                    {
                        'model_id': 1,
                        'model_name': 'LogisticRegression',
                        'ml_task': 'classification'
                    }
                }
            ]
        }
    }

class ModelFit(BaseModel):
    data_train : dict
    params : dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'data_train': 
                    {
                        'Name': 
                        {
                            0: 'Tom', 
                            1: 'Joseph', 
                            2: 'Krish', 
                            3: 'John'
                        },
                        'Age': 
                        {
                            0: 20, 
                            1: 21, 
                            2: 19, 
                            3: 18
                        }
                    },
                    'params': 
                    {
                        'random_state': 32
                    }
                }
            ]
        }
    }

class ModelPredict(BaseModel):
    data_test : dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'data_test': 
                    {
                        'Name': 
                        {
                            0: 'Marta', 
                            1: 'Den'
                        },
                        'Age': 
                        {
                            0: 22, 
                            1: 17
                        }
                    }
                }
            ]
        }
    }


@app.get("/api/model_list/{task}")
async def model_list(response: Response, task: str):
    """
    Получает на вход тип задачи и выводит список моделей доступных для ее решения \n
        task: тип задачи \n

    http://127.0.0.1:8000/api/model_list/{task}
    """
    result, response.status_code = models.available_model_list(task)
    return result

@app.get("/api/get_models")
async def get_all_models():
    """
    Возвращает список всех моделей \n
    http://127.0.0.1:8000/api/get_models \n
    """
    return models.models

@app.get("/api/get_model_by_id/{model_id}")
def get_model_by_id(response: Response, model_id: int):
    """
    Получает на вход id модели и возвращает ее \n
        model_id: id модели \n
    http://127.0.0.1:8000/api/get_model_by_id/{model_id} \n
    """
    result, response.status_code = models.get_model_by_id(model_id)
    return result

@app.post("/api/create_model")
async def create_model(response: Response, request: ModelItem):
    """
    Получает на вход название модели и создает модель  \n
        { \n
            "model_name": название модели, которое выбирает пользователь \n
        } \n
        return: { \n
            'model_id' - id модели \n
            'model_name' - название модели \n
            'ml_task' -  тип задачи \n
        } \n
    http://127.0.0.1:8000/api/create_model
    """
    result, response.status_code = models.create_model(model_name=request.model_name)
    return result

@app.put("/api/update_model")
async def update_model(response: Response, request: ModelUpd, status_code=status.HTTP_200_OK):
    """
    Получает на вход dict модели и обновляет его \n
        { \n
            "model_name":  \n
                {
                    "ml_task": "classification", \n
                    "model_id": 1, \n
                    "model_name": "LogisticRegression" \n
                } \n
        } \n
    http://127.0.0.1:8000/api/update_model
    """
    status_code = models.update_model(model_name=request.model_name)
    if (status_code != 200):
        response.status_code = status_code

@app.delete("/api/delete_model/{model_id}")
def delete_model(response: Response, model_id: int):
    """
    Получает на вход id и удаляет выбранную модель  \n
        model_id: id модели \n
    http://127.0.0.1:8000/api/delete_model/{model_id}
    """
    status_code = models.delete_model(model_id)
    if (status_code != 200):
        response.status_code = status_code

@app.put("/api/fit/{model_id}")
async def fit(response: Response, model_id: int, request: ModelFit):
    """
    Получает на вход id модели, данные для обучения и параметры, возвращает обученную модель \n
        model_id: id модели, \n
        data: данные (data_train и target) (dict) \n
        params: параметры для обучения (dict) \n
    http://127.0.0.1:8000/api/fit/{model_id}
    """
    model_id = model_id
    result, response.status_code = models.fit(model_id, request.data_train, request.params)
    return result

@app.put("/api/predict/{model_id}")
async def predict(response: Response, model_id: int, request: ModelPredict):
    """
    Получает на вход id модели и тестовую выборку, возвращает прогноз \n
        model_id: id модели, \n
        X: выборка для предсказания, без таргета (dict) \n
    http://127.0.0.1:8000/api/predict/{model_id}
    """
    result, response.status_code = models.predict(model_id, request.data_test)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)