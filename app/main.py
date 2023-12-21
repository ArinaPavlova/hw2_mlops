import pandas as pd
import logging
import io

import pickle
import json

from minio import Minio

from pandas.core.frame import DataFrame

from collections import defaultdict

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from typing import Dict, Union, Any

import uvicorn
from fastapi import FastAPI, status, Response
#from models import Models
from pydantic import BaseModel

app = FastAPI()

AVAILABLE_MODEL_LIST = {'regression': [Ridge, Lasso], 'classification': [LogisticRegression, GradientBoostingClassifier]}

def create_minio():
    minio_client = Minio(
        f"minio:9000",
        access_key='2aMFSfJf0Ar6bTjlLj48',
        secret_key='bI3l3Thl2PLEA9eRcZeLjnHlePq2iwVw47l1p3iP',
        secure=False
    )

    if not minio_client.bucket_exists('models'):
        minio_client.make_bucket('models')

    if not minio_client.bucket_exists('fitted-models'):
        minio_client.make_bucket('fitted-models')
    return minio_client


def insert_json_in_db(model_dict, minio_client):
    model_id = model_dict['model_id']
    model_id_str = str(model_id)
    bytes_file = json.dumps(model_dict).encode()
    minio_client.put_object('models', f'{model_id_str}.json', data=io.BytesIO(bytes_file),
                                        length=len(bytes_file))


def insert_model_in_db(model_dict, minio_client):
    model_id = model_dict['model_id']
    model_id_str = str(model_id)
    bytes_file = pickle.dumps(model_dict['model'])
    minio_client.put_object('fitted-models', f'{model_id_str}.pkl', data=io.BytesIO(bytes_file), length=len(bytes_file))


def del_model_from_minio(model_id: int, minio_client, model=False):
    minio_client.remove_object('models', f'{model_id}.json')
    if model:
        minio_client.remove_object('fitted-models', f'{model_id}.pkl')

class Models:
    def __init__(self):
        self.counter = 0
        self.ml_task = None
        self.available_models = defaultdict()
        self.minio_client = create_minio()

    def available_model_list(self, task: str = '') -> str:
        """
        Получает на вход тип задачи и выводит список моделей доступных для ее решения 

        task: тип задачи
        """
        if task not in ['regression', 'classification']:
            logging.error(f"Invalid task type '{task}'. Available task types: 'regression', 'classification'")
            return "Invalid task type. Available task types: 'regression', 'classification'", 400  # Bad request
        self.ml_task = task
        self.available_models[self.ml_task] = {md.__name__: md for md in AVAILABLE_MODEL_LIST[self.ml_task]}
        to_print = [md.__name__ for md in AVAILABLE_MODEL_LIST[self.ml_task]]
        return f"ML task '{self.ml_task}':    Models: {to_print}", 200

    def get_model_by_id(self, model_id: int, model_return = False) -> Dict:
        """
        Получает на вход id модели и возвращает ее

        model_id: id модели
        fitted: указывает, нужно ли получить подготовленную модель (True) или необученную модель (False).
        """
        try:
            if model_return:
                response = self.minio_client.get_object('fitted-models', f'{model_id}.pkl')
                model = pickle.loads(response.data)
            else:
                response = self.minio_client.get_object('models', f'{model_id}.json')
                model = json.loads(response.data)
            return model, 200   
        except:
            logging.error(f"ML model {model_id} doesn't exist")
            return "ML model doesn't exist", 404  # Not found
    

    def create_model(self, model_name: str = '') -> Dict:
        """
        Получает на вход название модели и создает модель 

        model_name: название модели, которое выбирает пользователь

        return: {
            'model_id' - id модели
            'model_name' - название модели
            'ml_task' -  тип задачи
        }
        """
        self.counter += 1
        ml_model = {
            'model_id': self.counter,
            'model_name': None,
            'ml_task': self.ml_task,
        }

        if model_name in self.available_models[self.ml_task]:
            ml_model['model_name'] = model_name
        else:
            self.counter -= 1
            logging.error(f"Wrong model name {model_name}. Available models: {list(self.available_models[self.ml_task].keys())}")
            return "Wrong model name", 400  # Bad request
        
        insert_json_in_db(ml_model, self.minio_client)
        return ml_model, 201
    
    
    def update_model(self, model_dict: dict) -> None:
        """
        Получает на вход dict модели и обновляет его

        model_dict: dict модели
        """
        try:
            insert_json_in_db(model_dict, self.minio_client)
            return 200
        except (KeyError, TypeError):
            logging.error("Incorrect dictionary passed. Dictionary should be passed.")
            return 400  # Bad request

    def delete_model(self, model_id: int) -> None:
        """
        Получает на вход id и удаляет выбранную модель 

        model_id: id модели
        """
        try:
            del_model_from_minio(model_id, self.minio_client, model = True)
            return 200
        except ValueError:
            logging.error(f"ML model {model_id} doesn't exist")
            return 404  # Not found

    def fit(self, model_id, data_train, params) -> Dict:
        """
        Получает на вход id модели, данные для обучения и параметры, возвращает обученную модель

        model_id: id модели,
        data: данные (data_train и target)
        params: параметры для обучения
        """
        try:
            target = pd.DataFrame(data_train)[['target']]
            data_train = pd.DataFrame(data_train).drop(columns='target') 
        except Exception as e:
            logging.error(f"An error with input data: {e}")
            return "An error occurred with input data", 400  # Bad request
        
        try:
            model_dict = self.get_model_by_id(model_id)[0]
            fitted_model = self.get_model_by_id(model_id)[0]
        except ValueError:
            logging.error(f"ML model {model_id} doesn't exist")
            return "ML model doesn't exist", 404  # Not found
        
        try:
            ml_mod = self.available_models[self.ml_task][model_dict['model_name']](**params)
        except TypeError:
            logging.error(f"Incorrect model parameters {params}.")
            return "Incorrect model parameters", 400  # Bad request
        
        try:
            ml_mod.fit(data_train, target)
        except Exception as e:
            logging.error(f"An error occurred during fitting: {e}")
            return "An error occurred during fitting", 500  # Internal server error
        
        try:
            fitted_model['model'] = ml_mod
            insert_model_in_db(fitted_model, self.minio_client)
            return model_dict, 200
        except Exception as e:
            logging.error(f"Something wrong: {e}")
            return "Something wrong", 500  # Internal server error
        


    def predict(self, model_id, data_test) -> Union[DataFrame, Any]:
        """
        Получает на вход id модели и тестовую выборку, возвращает прогноз

        model_id: id модели,
        X: выборка для предсказания, без таргета
        """
        try:
            data_test = pd.DataFrame(data_test)
        except Exception as e:
            logging.error(f"An error with input data: {e}")
            return "An error occurred with input data", 400  # Bad request
        
        try:
            model = self.get_model_by_id(model_id, model_return = True)[0]
        except ValueError:
            logging.error(f"ML model {model_id} doesn't exist")
            return "ML model doesn't exist", 404  # Not found
        except Exception as e:
                logging.error(f"Something wrong: {e}")
                return "Something wrong", 500  # Internal server error
        
        try:
            predict = model.predict(data_test)
            return pd.DataFrame(predict).to_dict(), 200
        except Exception as e:
                logging.error(f"An error occurred during prediction: {e}")
                return "An error occurred during prediction", 500  # Internal server error

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
    result, response.status_code = Models().available_model_list(task)
    return result

@app.get("/api/get_models")
async def get_all_models():
    """
    Возвращает список всех моделей \n
    http://127.0.0.1:8000/api/get_models \n
    """
    return Models().models

@app.get("/api/get_model_by_id/{model_id}")
def get_model_by_id(response: Response, model_id: int):
    """
    Получает на вход id модели и возвращает ее \n
        model_id: id модели \n
    http://127.0.0.1:8000/api/get_model_by_id/{model_id} \n
    """
    result, response.status_code = Models().get_model_by_id(model_id)
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
    result, response.status_code = Models().create_model(model_name=request.model_name)
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
    status_code = Models().update_model(model_name=request.model_name)
    if (status_code != 200):
        response.status_code = status_code

@app.delete("/api/delete_model/{model_id}")
def delete_model(response: Response, model_id: int):
    """
    Получает на вход id и удаляет выбранную модель  \n
        model_id: id модели \n
    http://127.0.0.1:8000/api/delete_model/{model_id}
    """
    status_code = Models().delete_model(model_id)
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
    result, response.status_code = Models().fit(model_id, request.data_train, request.params)
    return result

@app.put("/api/predict/{model_id}")
async def predict(response: Response, model_id: int, request: ModelPredict):
    """
    Получает на вход id модели и тестовую выборку, возвращает прогноз \n
        model_id: id модели, \n
        X: выборка для предсказания, без таргета (dict) \n
    http://127.0.0.1:8000/api/predict/{model_id}
    """
    result, response.status_code = Models().predict(model_id, request.data_test)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)