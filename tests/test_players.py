# -*- encoding: utf-8 -*-

from io import StringIO
import unittest
from bson.binary import Binary as BsonBinary
from players import *
import pandas as pd

class TestPlayers(unittest.TestCase):

    @classmethod
    def __initialize(cls):
        cls.con[PROJECTS_COLLECTION_NAME].drop()
        cls.con[COLUMNS_COLLECTION_NAME].drop()
        cls.con[ALGORITHMS_COLLECTION_NAME].drop()
        cls.con[ANALYSES_COLLECTION_NAME].drop()

    @classmethod
    def __prepare_mongodb_db(cls, file_name):
        with open(file_name, 'r') as f:
            header = f.readline().split(',')
            mongo_data = []
            for row in f.readlines():
                mongo_row_data = {}
                row_clumns = row.split(',')
                for column, r in zip(header, row_clumns):
                    mongo_row_data[column.strip()] = r.strip()
            mongo_data.append(mongo_row_data)
        return mongo_data


    @classmethod
    def __prepare_analses(cls):
        projects = list(cls.con[PROJECTS_COLLECTION_NAME].find())
        algorithms = list(cls.con[ALGORITHMS_COLLECTION_NAME].find())
        cursor = cls.con[PROJECTS_COLLECTION_NAME].find()
        data = []
        for p, a in zip(projects, algorithms):
            data.append({'project_id': p['_id'], 'algorithm_id': a['_id']})

        cls.con[ANALYSES_COLLECTION_NAME].insert_many(data)

    @classmethod
    def __prepare_projects(cls):
        cls.con[PROJECTS_COLLECTION_NAME].insert_many(cls.__prepare_mongodb_db('./tests/data/projects.csv'))
        cursor = cls.con[PROJECTS_COLLECTION_NAME].find()
        for c in cursor:
            file_name = c['file_name']
            with open(file_name, 'rb') as f:
                cls.con[PROJECTS_COLLECTION_NAME].update_one({'_id': c['_id']}, {"$set": {'file': BsonBinary(f.read())}})


    @classmethod
    def __prepare_columns(cls):
        cursor = cls.con[PROJECTS_COLLECTION_NAME].find()
        data = []
        for c in cursor:
            with open(c['file_name'], 'r') as f:
                header = f.readline().split(',')
            # consider last columns is target in this test
            train_columns = header[:-1]
            target_columns = header[-1]
            data.append({'project_id': c['_id'], 'train_columns': ",".join(train_columns), 'target_columns': target_columns})

        cls.con[COLUMNS_COLLECTION_NAME].insert_many(data)


    @classmethod
    def __prepare_algorithms(cls):
        cls.con[ALGORITHMS_COLLECTION_NAME].insert_many(cls.__prepare_mongodb_db('./tests/data/algorithms.csv'))


    @classmethod
    def setUpClass(cls):
        cls.config = parse_config('development')
        cls.con = db(cls.config)
        cls.__initialize()
        cls.__prepare_projects()
        cls.__prepare_columns()
        cls.__prepare_algorithms()
        cls.__prepare_analses()


    @classmethod
    def tearDownClass(cls):
        # make sure no data remails
        print(cls.con[ALGORITHMS_COLLECTION_NAME].drop())
        print(list(cls.con[ALGORITHMS_COLLECTION_NAME].find()))
        print(cls.con[PROJECTS_COLLECTION_NAME].drop())
        print(list(cls.con[PROJECTS_COLLECTION_NAME].find()))


    def setUp(self):
        print("setup")

    def test_player(self):
        print("test")

    def test_dispatcher(self):
        d = Dispacher(self.con)
        d.dispatch()


    """
    def test_player(self):
        print("test_players start")
        algorithm = list(self.con[ALGORITHM_COLLECTION_NAME].find())[0]
        project = list(self.con[PROJECT_COLLECTION_NAME].find())[0]
        data = project['file']
        df = pd.read_csv(StringIO(BsonBinary(data).decode()))
        player = Player(algorithm['module_name'], algorithm['class_name'], df.loc[:, 'A':'D'], df.loc[:, 'E'])
        player.play()
    """

if __name__ == '__main__':
    unittest.main()
