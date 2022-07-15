from __future__ import annotations

import os
from configparser import ConfigParser
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, requests
from pandas import read_json, DataFrame
import json
import requests

from rex.dump import decode_model, encode_model, dump_model, load_model
from rex.model import Rex, RexBaseModel
from rex.preprocessing import Bin, PreprocessPipeline, PreprocessedDataFrame, Drop, PreprocessFunction
from rex.tools import get_df

CLIENT_CONFIGS = '../../resources/client_config.ini'
SERVER_CONFIGS = '../resources/server_config.ini'
OK_MESSAGE = {'message': 'ok'}
UNPROCESSABLE_ENTITY = 422
app = FastAPI()


def _load_server_config():
    configs = ConfigParser()
    configs.read(SERVER_CONFIGS)
    return dict(configs.items('server_configurations'))


def _load_client_config():
    configs = ConfigParser()
    configs.read(CLIENT_CONFIGS)
    return dict(configs.items('client_configurations'))


def _raise_bad_formatted_data(message: str):
    raise HTTPException(status_code=UNPROCESSABLE_ENTITY, detail=message)


def _make_folder_if_not(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def _make_path(folder_name: str):
    config = _load_server_config()
    return f"{config['data_path']}/{config[folder_name]}"


def _send(route: str, data: dict) -> any:
    configs = _load_client_config()
    server = f"{configs['host']}:{configs['port']}{route}"
    result = requests.post(server, json=data)
    if result.ok:
        return json.loads(result.text)
    else:
        print(result.text)
        return None


def remote_preprocess(dataframe_path: str,
                      dataframe_file_name: str,
                      preprocess_functions: List[PreprocessFunction]) -> Optional[DataFrame]:
    preprocess_functions_json = [{'name': type(function).__name__, 'parameters': function.__dict__}
                                 for function in preprocess_functions]
    data = {'dataframe_path': dataframe_path,
            'dataframe_file_name': dataframe_file_name,
            'preprocess_functions': preprocess_functions_json}
    return _send(_load_client_config()['preprocess_route'], data)


def remote_fit(dataset_file_name: str,
               model_file_name: str,
               model_parameters: Optional[dict] = None,
               fit_parameters: Optional[dict] = None) -> Optional[str]:
    data = {'dataset_file_name': dataset_file_name,
            'model_file_name': model_file_name,
            'model_parameters': model_parameters,
            'fit_parameters': fit_parameters}
    return _send(_load_client_config()['fit_route'], data)


def remote_predict(model_file_name: str,
                   predictions_file_name: str,
                   user_ids: List[any],
                   item_ids: List[any],
                   predict_parameters: Optional[dict] = None) -> Optional[DataFrame]:
    data = {'model_file_name': model_file_name,
            'predictions_file_name': predictions_file_name,
            'user_ids': np.array(user_ids).tolist(),
            'item_ids': np.array(item_ids).tolist(),
            'predict_parameters': predict_parameters}
    return _send(_load_client_config()['predict_route'], data)


def remote_fit_predict(dataset_file_name: str,
                       model_file_name: str,
                       predictions_file_name: str,
                       user_ids: List[any],
                       item_ids: List[any],
                       model_parameters: Optional[dict] = None,
                       fit_parameters: Optional[dict] = None,
                       predict_parameters: Optional[dict] = None):
    fit_result = remote_fit(dataset_file_name, model_file_name, model_parameters, fit_parameters)
    return remote_predict(model_file_name, predictions_file_name, user_ids, item_ids, predict_parameters) \
        if (fit_result is not None) \
        else None


# ----------------------- REMOTE CALLS -----------------------
@app.post('/preprocess')
async def preprocess(dataset_and_functions: Request):
    data = await dataset_and_functions.json()

    # check values
    if 'dataframe_path' not in data or 'preprocess_functions' not in data or 'dataframe_file_name' not in data:
        _raise_bad_formatted_data("json must contains 'dataframe_path', "
                                  "'dataframe_file_name' and "
                                  "'preprocess_functions'")

    try:
        # create folder if it doesn't exist
        dataset_path = _make_path('dataset_folder')
        _make_folder_if_not(dataset_path)

        # load dataframe
        dataframe = pd.read_csv(data['dataframe_path'])

        # check functions
        if not isinstance(data['preprocess_functions'], list):
            _raise_bad_formatted_data("'preprocess_functions' must be a list")

        # create preprocess functions
        preprocess_functions = []
        for preprocess_function in data['preprocess_functions']:
            # check single function
            if 'name' not in preprocess_function or 'parameters' not in preprocess_function:
                _raise_bad_formatted_data("preprocess functions must have both 'name' and 'parameters' field")
            name = preprocess_function['name']
            parameters = preprocess_function['parameters']

            # TODO: complete
            real_function = None
            if name == 'Bin':
                real_function = Bin
            elif name == 'Drop':
                real_function = Drop
            else:
                _raise_bad_formatted_data(f"Invalid function name '{name}'")
            preprocess_functions.append(real_function(**parameters))

        # apply all preprocess functions and extract, incase, dataframe
        preprocessed_dataframe = get_df(PreprocessPipeline(preprocess_functions).fit_transform(dataframe))
        preprocessed_dataframe.to_csv(f"{dataset_path}/{data['dataframe_file_name']}", index=False)

    except Exception as e:
        _raise_bad_formatted_data(str(e))

    return OK_MESSAGE


@app.post('/fit_predict')
async def fit_transform(dataset_model_model_parameters: Request):
    await fit(dataset_model_model_parameters)
    return await predict(dataset_model_model_parameters)


@app.post('/fit')
async def fit(dataset_model_parameters: Request):
    data = await dataset_model_parameters.json()

    # check values
    if 'dataset_file_name' not in data or \
            'model_file_name' not in data or \
            'model_parameters' not in data or \
            'fit_parameters' not in data:
        _raise_bad_formatted_data("json must contains 'dataset_file_name', "
                                  "optionally 'model_parameters' and "
                                  "'fit_parameters'")
    try:
        # create folder if it doesn't exist
        fit_path = _make_path('fit_folder')
        dataset_path = _make_path('dataset_folder')
        _make_folder_if_not(fit_path)

        # load dataframe
        dataset = pd.read_csv(f"{dataset_path}/{data['dataset_file_name']}")
        # parameters
        model_parameters = {} if data['model_parameters'] is None else data['model_parameters']
        fit_parameters = {} if data['fit_parameters'] is None else data['fit_parameters']
        # create model
        model = Rex(**model_parameters)
        # fit model
        model.fit(dataset, **fit_parameters)
        # save model
        dump_model(model, f"{fit_path}/{data['model_file_name']}")
    except Exception as e:
        _raise_bad_formatted_data(str(e))

    return OK_MESSAGE


@app.post('/predict')
async def predict(predictions_model_predict_parameters: Request):
    data = await predictions_model_predict_parameters.json()
    # check values
    if 'user_ids' not in data or \
            'item_ids' not in data or \
            'model_file_name' not in data or \
            'predictions_file_name' not in data:
        _raise_bad_formatted_data("json must contains 'mode_file_name', 'predictions_file_name', "
                                  "'user_ids', 'item_ids' and optionally 'predict_parameters'")
    try:
        # create folder if it doesn't exist
        fit_path = _make_path('fit_folder')
        predictions_path = _make_path('predictions_folder')
        _make_folder_if_not(predictions_path)
        # load model
        model = load_model(f"{fit_path}/{data['model_file_name']}")

        # check ids
        if not isinstance(data['user_ids'], list) or not isinstance(data['item_ids'], list):
            _raise_bad_formatted_data("'user_ids' and 'item_ids' must be a list")

        # predict params
        predict_params = {} if data['predict_parameters'] is None else data['predict_parameters']
        # compute predictions
        predictions = model.predict(data['user_ids'], data['item_ids'], **predict_params)
        # save predictions
        predictions.to_csv(f"{predictions_path}/{data['predictions_file_name']}", index=False)

    except Exception as e:
        _raise_bad_formatted_data(f"Error: {str(e)}")

    return OK_MESSAGE
