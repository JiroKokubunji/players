# -*- encoding: utf-8 -*-

import unittest
from players import *

class TestPlayers(unittest.TestCase):

    @classmethod
    def __initialize(cls):
        cls.con[PROJECTS_COLLECTION_NAME].drop()
        cls.con[COLUMNS_COLLECTION_NAME].drop()
        cls.con[ALGORITHMS_COLLECTION_NAME].drop()
        cls.con[ANALYSES_COLLECTION_NAME].drop()
        cls.con[CLASSIFICATION_RESULTS_COLLECTION_NAME].drop()
        cls.con[PREPROCESSED_DATA_COLLECTION_NAME].drop()

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
        preprocessed_data = list(cls.con[PREPROCESSED_DATA_COLLECTION_NAME].find())
        algorithms = list(cls.con[ALGORITHMS_COLLECTION_NAME].find())
        data = [{'preprocessed_data_id': p['_id'], 'algorithm_id': a['_id']} for p, a in zip(preprocessed_data, algorithms)]
        cls.con[ANALYSES_COLLECTION_NAME].insert_many(data)

    @classmethod
    def __prepare_projects(cls):
        # prepare project data, it has a multiple processed data, first one is the original data.
        cls.con[PROJECTS_COLLECTION_NAME].insert_many(cls.__prepare_mongodb_db('./tests/data/projects.csv'))
        cursor = cls.con[PROJECTS_COLLECTION_NAME].find()
        for c in cursor:
            file_name = c['file_name']
            with open(file_name, 'r') as f:
                file = f.read()
                cls.con[PROJECTS_COLLECTION_NAME].update_one({'_id': c['_id']}, {"$set": {'file': file}})
            # create first processed data, first one is not processed exactly.
            result = cls.con[PREPROCESSED_DATA_COLLECTION_NAME].insert_one({'project_id': c['_id'], 'data': file})
            # create columns data of first processed data
            header = StringIO(file).readline().strip().split(',')
            # consider last columns is target in this test
            train_header = [{'preprocessed_data_id': result.inserted_id, 'name': th, 'target': False} for th in header[:-1]]
            target_header = [{'preprocessed_data_id': result.inserted_id, 'name': th, 'target': True} for th in header[-1]]
            train_header.extend(target_header)
            cls.con[COLUMNS_COLLECTION_NAME].insert_many(train_header)

    """
    @classmethod
    def __prepare_columns(cls):
        cursor = cls.con[PREPROCESSED_DATA_COLLECTION_NAME].find()
        data = []
        for c in cursor:
            project = cls.con[PROJECTS_COLLECTION_NAME].find_one({'_id': c['project_id']})
            with open(project['file_name'], 'r') as f:
                header = f.readline().split(',')
            # consider last columns is target in this test
            train_columns = header[:-1]
            target_columns = header[-1]
            for train in train_columns:
                data.append({'preprocessed_data_id': c['_id'], 'name': train, 'target': False})
            for target in target_columns:
                data.append({'preprocessed_data_id': c['_id'], 'name': target, 'target': True})

        cls.con[COLUMNS_COLLECTION_NAME].insert_many(data)
    """

    @classmethod
    def __prepare_algorithms(cls):
        cls.con[ALGORITHMS_COLLECTION_NAME].insert_many(cls.__prepare_mongodb_db('./tests/data/algorithms.csv'))


    @classmethod
    def __prepare_preprocesses(cls):
        cursor = cls.con[PREPROCESSED_DATA_COLLECTION_NAME].find()
        for c in cursor:
            cls.con[PREPROCESS_ORDER_COLLECTION_NAME].insert_one({'preprocessed_data_id': c['_id'], 'type': 'LabelEncoder', 'column': 'A', 'order': 1})
            cls.con[PREPROCESS_ORDER_COLLECTION_NAME].insert_one({'preprocessed_data_id': c['_id'], 'type': 'OneHotEncoder', 'column': 'A.0', 'order': 2})


    @classmethod
    def setUpClass(cls):
        cls.config = parse_config('development')
        cls.con = db(cls.config)
        cls.__initialize()
        cls.__prepare_algorithms()
        cls.__prepare_projects()
        cls.__prepare_analses()
        cls.__prepare_preprocesses()
        # cls.__prepare_columns()


    @classmethod
    def tearDownClass(cls):
        # make sure no data remains
        cls.con[PREPROCESS_ORDER_COLLECTION_NAME].drop()
        cls.con[PREPROCESSED_DATA_COLLECTION_NAME].drop()
        cls.con[CLASSIFICATION_RESULTS_COLLECTION_NAME].drop()
        cls.con[ANALYSES_COLLECTION_NAME].drop()
        cls.con[ALGORITHMS_COLLECTION_NAME].drop()
        cls.con[COLUMNS_COLLECTION_NAME].drop()
        cls.con[PROJECTS_COLLECTION_NAME].drop()

    def setUp(self):
        pass

    def test_00_preprocess(self):
        print("test preprocess")
        ppds = self.con[PREPROCESSED_DATA_COLLECTION_NAME].find()
        for ppd in ppds:
            p = Preprocessor(self.con, ppd['_id'])
            p.preprocess()

        upd = self.con[PREPROCESSED_DATA_COLLECTION_NAME].find()
        upd = upd[0]
        df = pd.read_csv(StringIO(upd['data']))
        df.drop('A', axis=1, inplace=True)
        data_buf = StringIO()
        df.to_csv(data_buf, index=False)
        self.con[PREPROCESSED_DATA_COLLECTION_NAME].update_one({'_id': upd['_id']}, {"$set" : { "data": data_buf.getvalue()}})
        c = self.con[COLUMNS_COLLECTION_NAME].find_one({'preprocessed_data_id': upd['_id']})
        self.con[COLUMNS_COLLECTION_NAME].delete_one({'_id': c['_id'], 'name': 'A'})


    def test_01_dispatcher(self):
        print("test_dispatcher")
        d = Dispacher(self.con)
        d.dispatch()
        # check results
        # print(list(self.con[CLASSIFICATION_RESULTS_COLLECTION_NAME].find()))

    def test_02_player(self):
        print("test_players start")
        analysis = list(self.con[ANALYSES_COLLECTION_NAME].find())[0]
        algorithm = self.con[ALGORITHMS_COLLECTION_NAME].find_one({'_id': analysis['algorithm_id']})
        preprocess_data = self.con[PREPROCESSED_DATA_COLLECTION_NAME].find_one({'_id': analysis['preprocessed_data_id']})
        data = preprocess_data['data']
        df = pd.read_csv(StringIO(data))
        player = Player(analysis['_id'], algorithm['module_name'], algorithm['class_name'], data, df.drop('E', axis=1).columns, ['E'])
        player.play()

if __name__ == '__main__':
    unittest.main()
