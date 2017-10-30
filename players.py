# -*- encoding: utf-8 -*-

import importlib
import argparse
import time
import yaml
from pymongo import MongoClient
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ANALYZE_COLLECTION_NAME = 'analyses'
PROJECT_COLLECTION_NAME = 'projects'
MODELS_COLLECTION = 'models'
ALGORITHM_COLLECTION_NAME = 'algorithms'
NUMBER_OF_MAX_PROCESS = 4

class Dispacher:
    """
    dispathcer of playsers, dispatch tasks to players
    """

    def __init__(self, db):
        self.db = db

    def dispatch(self):
        while True:
            players = self._prepare_players()
            pool = Pool(processes=4)
            results = pool.map(wrap_players, players)
            print("results:{0}".format(results))
            pool.close()
            print('process ended')
            time.sleep(1)
            break

    def _prepare_players(self):
        todos = self.db[ANALYZE_COLLECTION_NAME].find()
        data_for_players = []
        for todo in todos:
            data_for_players.append(self._get_project_data(todo['project_id']))

        players = []
        for dfp in data_for_players:
            model = self.db[MODELS_COLLECTION].find_one({'project_id': dfp['_id']})
            algorithm = self.db[ALGORITHM_COLLECTION_NAME].find_one({'id': dfp['algorithm_id']})
            players.append(Player(algorithm['module_name'], algorithm['class_name'], dfp['data'], dfp['data']))
        return players

    def _get_project_data(self, project_id):
        return self.db[PROJECT_COLLECTION_NAME].find_one({'_id': project_id})


class Player:

    TEST_SIZE = 0.33

    def __init__(self, package_name, class_name, data, target):
        self.package_name = package_name
        self.class_name = class_name
        self.data = data
        self.target = target

    def play(self):
        module = importlib.import_module(self.package_name)
        instance = getattr(module, self.class_name)()
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=self.TEST_SIZE)
        instance.fit(X_train, y_train)
        y_pred = instance.predict(X_test)
        return accuracy_score(y_test, y_pred)


def wrap_players(args):
    return Player(args[0], args[1], args[2], args[3]).play()


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
