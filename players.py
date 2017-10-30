# -*- encoding: utf-8 -*-

import importlib
import argparse
import time
import yaml
from pymongo import MongoClient
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from io import StringIO
from bson.binary import Binary as BsonBinary

ANALYSES_COLLECTION_NAME = 'analyses'
PROJECTS_COLLECTION_NAME = 'projects'
COLUMNS_COLLECTION_NAME = 'columns'
MODELS_COLLECTION = 'models'
ALGORITHMS_COLLECTION_NAME = 'algorithms'
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
            print("results:{0}".format(results))
            pool.close()
            print('process ended')
            time.sleep(1)
            break

    def __prepare_players(self):
        todos = self.db[ANALYSES_COLLECTION_NAME].find()
        players = []
        for todo in todos:
            project = self.__get_project_data(todo['project_id'])
            columns = self.db[COLUMNS_COLLECTION_NAME].find_one({'project_id': todo['project_id']})
            algorithm = self.db[ALGORITHMS_COLLECTION_NAME].find_one({'_id': todo['algorithm_id']})
            data = StringIO(BsonBinary(project['file']).decode())
            players.append(Player(algorithm['module_name']
                                  , algorithm['class_name']
                                  , data
                                  , columns['train_columns']
                                  , columns['target_columns']))
        return players

    def __get_project_data(self, project_id):
        return self.db[PROJECTS_COLLECTION_NAME].find_one({'_id': project_id})


class Player:

    TEST_SIZE = 0.33

    def __init__(self, package_name, class_name, data, train_columns, target_columns):
        self.package_name = package_name
        self.class_name = class_name
        self.data = data
        self.train_columns = train_columns
        self.target_columns = target_columns


    def play(self):
        module = importlib.import_module(self.package_name)
        instance = getattr(module, self.class_name)()
        df = pd.read_csv(self.data)
        train = df.loc[:, self.train_columns.split(',')]
        target = df.loc[:, self.target_columns.strip()]
        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=self.TEST_SIZE)
        instance.fit(X_train, y_train)
        y_pred = instance.predict(X_test)
        return accuracy_score(y_test, y_pred)


def wrap_players(args):
    return args.play()
    # return Player(args[0], args[1], args[2], args[3], args[4]).play()


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
