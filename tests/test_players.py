# -*- encoding: utf-8 -*-

from io import StringIO
import unittest
from bson.binary import Binary as BsonBinary
from players import *
import pandas as pd

class TestPlayers(unittest.TestCase):

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
    def __prepare_projects(cls):
        cls.con[PROJECT_COLLECTION_NAME].insert_many(cls.__prepare_mongodb_db('./tests/data/projects.csv'))
        cursor = cls.con[PROJECT_COLLECTION_NAME].find()
        for c in cursor:
            file_name = c['file_name']
            with open(file_name, 'rb') as f:
                cls.con[PROJECT_COLLECTION_NAME].update_one({'_id': c['_id']}, {"$set": {'file': BsonBinary(f.read())}})

        print(list(cls.con[PROJECT_COLLECTION_NAME].find()))

    @classmethod
    def __prepare_algorithms(cls):
        cls.con[ALGORITHM_COLLECTION_NAME].insert_many(cls.__prepare_mongodb_db('./tests/data/algorithms.csv'))
        print(list(cls.con[ALGORITHM_COLLECTION_NAME].find()))

    @classmethod
    def setUpClass(cls):
        cls.config = parse_config('development')
        cls.con = db(cls.config)
        cls.__prepare_projects()
        cls.__prepare_algorithms()



    @classmethod
    def tearDownClass(cls):
        print(cls.con[ALGORITHM_COLLECTION_NAME].drop())
        print(list(cls.con[ALGORITHM_COLLECTION_NAME].find()))
        print(cls.con[PROJECT_COLLECTION_NAME].drop())
        print(list(cls.con[PROJECT_COLLECTION_NAME].find()))

    def setUp(self):
        print("setup")

    def test_player(self):
        print("test_players start")
        algorithm = list(self.con[ALGORITHM_COLLECTION_NAME].find())[0]
        project = list(self.con[PROJECT_COLLECTION_NAME].find())[0]
        data = project['file']
        df = pd.read_csv(StringIO(BsonBinary(data).decode()))
        player = Player(algorithm['module_name'], algorithm['class_name'], df.loc[:, 'A':'D'], df.loc[:, 'E'])
        player.play()

if __name__ == '__main__':
    unittest.main()
