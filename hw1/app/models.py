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

AVAILABLE_MODEL_LIST = {'regression': [Ridge, Lasso], 'classification': [LogisticRegression, GradientBoostingClassifier]}

class Models:
    def __init__(self):
        self.counter = 0
        self.ml_task = None
        self.available_models = defaultdict()
        self.minio_client = Minio(
                f"127.0.0.1:9000",
                access_key='2aMFSfJf0Ar6bTjlLj48',
                secret_key='bI3l3Thl2PLEA9eRcZeLjnHlePq2iwVw47l1p3iP',
                secure=False
                )

        if not self.minio_client.bucket_exists('models'):
            self.minio_client.make_bucket('models')

        if not self.minio_client.bucket_exists('fitted-models'):
            self.minio_client.make_bucket('fitted-models')

    def insert_json_in_db(self, model_dict):
        model_id = model_dict['model_id']
        model_id_str = str(model_id)
        bytes_file = json.dumps(model_dict).encode()
        self.minio_client.put_object('models', f'{model_id_str}.json', data=io.BytesIO(bytes_file), length=len(bytes_file))

    def insert_model_in_db(self, model_dict):
        model_id = model_dict['model_id']
        model_id_str = str(model_id)
        bytes_file = pickle.dumps(model_dict['model'])
        self.minio_client.put_object('fitted-models', f'{model_id_str}.pkl', data=io.BytesIO(bytes_file), length=len(bytes_file))

    def del_model_from_minio(self, model_id: int, model = False):
        self.minio_client.remove_object('models', f'{model_id}.json')
        if model:
            self.minio_client.remove_object('fitted-models', f'{model_id}.pkl')

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
        
        self.insert_json_in_db(ml_model)
        return ml_model, 201
    
    
    def update_model(self, model_dict: dict) -> None:
        """
        Получает на вход dict модели и обновляет его

        model_dict: dict модели
        """
        try:
            self.insert_json_in_db(self, model_dict)
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
            self.del_model_from_minio(model_id, model = True)
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
            self.insert_model_in_db(fitted_model)
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

