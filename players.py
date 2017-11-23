# -*- encoding: utf-8 -*-

import importlib
import argparse
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
from io import StringIO
from models.models import ProjectDatumColumns, \
                            TrainingRequests, \
                            TrainingRequestQueues, \
                            MachineLearningAlgorithms, \
    ClassificationTrainingResults
from bson.objectid import ObjectId

from db import setup_db

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
    """ dispathcer of playsers, dispatch tasks to players """

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
        ClassificationTrainingResults(
            training_request_id=result['training_request_id'],
            accuracy=result['accuracy'],
            recall=result['recall'],
            f1=result['f1'],
        ).save()
        trgr_queue = TrainingRequests.objects.raw({"_id": result['training_request_id']}).first()
        trgr_queue.status = "completed"
        trgr_queue.save()

    def __prepare_players(self):
        trgr_queues = TrainingRequestQueues.objects.raw({"status": "pendding"})
        for q in trgr_queues:
            # prepare data, and data
            trgr = q.training_request_id
            project_data = trgr.project_data_id
            data = project_data.data
            target_algorithms = list(map(lambda x: ObjectId(x), trgr.target_algorithms))
            algorithms = MachineLearningAlgorithms.objects.raw({'_id': {"$in" : target_algorithms}})
            # get valid columns and data
            trc = ProjectDatumColumns.objects.raw({
                "project_datum_id" : trgr.project_data_id._id,
                "active" : True,
                "target" : False
            })
            columns = list(map(lambda x: x.name, list(trc)))
            tgc = ProjectDatumColumns.objects.raw({
                "project_datum_id" : trgr.project_data_id._id,
                "active" : True,
                "target" : True
            }).first().name
            players = []
            for a in algorithms:
                 players.append(Player(trgr._id
                                      , a.module_name
                                      , a.class_name
                                      , data
                                      , columns
                                      , tgc))
        return players


class Player:

    TEST_SIZE = 0.33

    def __init__(self, training_request_id, package_name, class_name, data, train_columns, target_columns):
        self.training_request_id = training_request_id
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
        x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=self.TEST_SIZE)
        instance.fit(x_train, y_train)
        y_pred = instance.predict(x_test)
        return {"training_request_id": self.training_request_id
                    ,"accuracy": accuracy_score(y_test, y_pred)
                    , "recall": recall_score(y_test, y_pred)
                    , "f1": f1_score(y_test, y_pred)}


def wrap_players(args):
    return args.play()


def main(db):
    dispatcher = Dispacher(db)
    dispatcher.dispatch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The players is a app that run sklearn machine learning algorithms one by one. A data is provided by the Orchestra.')

    parser.add_argument('-e', '--environment', help='specify environment', default='development')
    args = parser.parse_args()
    environment = args.environment
    main(setup_db(environment))
