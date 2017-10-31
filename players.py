# -*- encoding: utf-8 -*-

import importlib
import argparse
import abc
import yaml
import pickle
from pymongo import MongoClient
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
from io import StringIO

ANALYSES_COLLECTION_NAME = 'analyses'
PROJECTS_COLLECTION_NAME = 'projects'
COLUMNS_COLLECTION_NAME = 'columns'
MODELS_COLLECTION = 'models'
ALGORITHMS_COLLECTION_NAME = 'algorithms'
CLASSIFICATION_RESULTS_COLLECTION_NAME = 'classification_results'
PREPROCESSED_DATA_COLLECTION_NAME = 'preprocess_data'
PREPROCESS_ORDER_COLLECTION_NAME = 'preprocess_order'
NUMBER_OF_MAX_PROCESS = 4

class Dispacher:
    """
    dispathcer of playsers, dispatch tasks to players
    """

    def __init__(self, db):
        self.db = db

    def dispatch(self):
        while True:
            players = self.__prepare_players()
            pool = Pool(processes=4)
            results = pool.map(wrap_players, players)
            for result in results:
                self.__record_result(result)
            pool.close()
            pool.join()
            break

    def __record_result(self, result):
        self.db[CLASSIFICATION_RESULTS_COLLECTION_NAME].insert_one(result)

    def __prepare_players(self):
        todos = self.db[ANALYSES_COLLECTION_NAME].find()
        players = []
        for todo in todos:
            preprocessed_data = self.db[PREPROCESSED_DATA_COLLECTION_NAME].find_one({'_id': todo['preprocessed_data_id']})
            columns = list(self.db[COLUMNS_COLLECTION_NAME].find({'preprocessed_data_id': preprocessed_data['_id']}))
            algorithm = self.db[ALGORITHMS_COLLECTION_NAME].find_one({'_id': todo['algorithm_id']})
            players.append(Player(todo['_id']
                                  , algorithm['module_name']
                                  , algorithm['class_name']
                                  , preprocessed_data['data']
                                  , [c['name'] for c in columns if not c['target']]
                                  , [c['name'] for c in columns if c['target']]))
        return players

    def __get_project_data(self, project_id):
        return self.db[PROJECTS_COLLECTION_NAME].find_one({'_id': project_id})


class IPreprocess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def do(self):
        pass


class OneHotEncoderPreProcessor(IPreprocess):
    def do(self, data):
        enc = OneHotEncoder()
        transformed = enc.fit_transform(data.reshape(data.reshape[0], -1))
        return enc, transformed.toarray()


class LabelEncoderPreProcessor(IPreprocess):
    def do(self, data):
        enc = LabelEncoder()
        return enc, enc.fit_transform(data.ravel())


class PreprocessorFactory:
    @classmethod
    def create(cls, type):
        if type == 'OneHotEncoder':
            return OneHotEncoderPreProcessor()
        elif type == 'LabelEncoder':
            return LabelEncoderPreProcessor()


class Preprocessor:

    def __init__(self, db, preprocessed_data_id):
        self.db = db
        self.preprocessed_data_id = preprocessed_data_id

    def preprocess(self):
        orders = self.db[PREPROCESS_ORDER_COLLECTION_NAME].find({'preprocessed_data_id': self.preprocessed_data_id})
        for order in orders:
            pp_data = self.db[PREPROCESSED_DATA_COLLECTION_NAME].find_one({'_id': self.preprocessed_data_id})
            df = pd.read_csv(StringIO(pp_data['data']))
            processor, processed_data = PreprocessorFactory.create(order['type']).do(df.loc[:, order['column']].values)
            columns_name = ["{0}.{1}".format(order['column'], c) for c in range(0, len(processed_data.shape))]
            df_1 = pd.DataFrame(processed_data, columns=columns_name)
            merged = pd.concat([df, df_1], axis=1)
            data_buf = StringIO()
            merged.to_csv(data_buf, index=False)
            self.db[PREPROCESS_ORDER_COLLECTION_NAME].update_one({'_id': order['_id']}, {"$set": {'processor': pickle.dumps(processor)}})
            self.db[PREPROCESSED_DATA_COLLECTION_NAME].update_one({'_id': pp_data['_id']}, {"$set": {'data': data_buf.getvalue()}})
            self.db[COLUMNS_COLLECTION_NAME].insert_many([{'preprocessed_data_id': self.preprocessed_data_id, 'name': c, 'target': False} for c in columns_name])


class Player:

    TEST_SIZE = 0.33

    def __init__(self, analyze_id, package_name, class_name, data, train_columns, target_columns):
        self.analyze_id = analyze_id
        self.package_name = package_name
        self.class_name = class_name
        self.data = data
        self.train_columns = train_columns
        self.target_columns = target_columns

    def play(self):
        module = importlib.import_module(self.package_name)
        instance = getattr(module, self.class_name)()
        df = pd.read_csv(StringIO(self.data))
        train = df.loc[:, self.train_columns]
        target = df.loc[:, self.target_columns]
        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=self.TEST_SIZE)
        instance.fit(X_train, y_train)
        y_pred = instance.predict(X_test)
        return {"analyze_id": self.analyze_id
                    ,"accuracy": accuracy_score(y_test, y_pred)
                    , "recall": recall_score(y_test, y_pred)
                    , "f1": f1_score(y_test, y_pred)}


def wrap_players(args):
    return args.play()


def main(db):
    dispatcher = Dispacher(db)
    dispatcher.dispatch()


def db(config):
    con = MongoClient("mongodb://{0}".format(config['clients']['default']['hosts'][0]))
    return con[config['clients']['default']['database']]


def parse_config(environment):
    with open('config/mongodb.yml') as f:
        config = yaml.load(f)
    return config[environment]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The players is a app that run sklearn machine learning algorithms one by one. A data is provided by the Orchestra.')

    parser.add_argument('-e', '--environment', help='specify environment', default='development')
    args = parser.parse_args()
    environment = args.environment
    config = parse_config(environment)
    main(db(config))
