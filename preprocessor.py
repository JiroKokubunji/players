# -*- coding: utf-8 -*-
from datetime import datetime
import abc
import argparse
from db import setup_db
from io import StringIO
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from models.models import ProcessColumnsRequestQueues, ProjectDatumColumns
from bson.objectid import ObjectId


class IPreprocess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def do(self):
        pass


class OneHotEncoderPreProcessor(IPreprocess):
    def do(self, data):
        enc = OneHotEncoder()
        transformed = enc.fit_transform(data.values.reshape(-1, 1))
        return enc, transformed.toarray()


class LabelEncoderPreProcessor(IPreprocess):
    def do(self, data):
        enc = LabelEncoder()
        return enc, enc.fit_transform(data.values.ravel())


class PreprocessorFactory:
    @classmethod
    def create(cls, type):
        if type == 'OneHotEncoder':
            return OneHotEncoderPreProcessor()
        elif type == 'LabelEncoder':
            return LabelEncoderPreProcessor()


class Preprocessor:
    def preprocess(self):
        while True:
            pcr_queues = ProcessColumnsRequestQueues.objects.raw({"status": "pendding"})
            for q in pcr_queues:
                pcr = q.process_columns_request_id
                project_data = pcr.project_data_id
                data = project_data.data
                df = pd.read_csv(StringIO(data))
                if pcr.task is None:
                    types = df.dtypes.to_dict()
                    for column_name, type in types.items():
                        column = ProjectDatumColumns.objects.raw({
                            'project_datum_id': pcr.project_data_id._id,
                            'name': column_name
                        }).first()
                        column.type = str(type)
                        column.save()
                else:
                    tc = list(map(lambda x: ObjectId(x), pcr.target_columns))
                    columns = ProjectDatumColumns.objects.raw({'_id': {"$in" : tc}})
                    p = PreprocessorFactory.create(pcr.task)
                    n = list(map(lambda x: x.name, list(columns)))
                    p, processed_data = p.do(df.loc[:, n].dropna())
                    column_num = processed_data.shape[1] if len(processed_data.shape) > 1 else 1
                    columns_name = ["{0}.{1}".format(n, c) for c in range(0, column_num)]
                    df_1 = pd.DataFrame(processed_data, columns=columns_name)
                    column_types = df_1.dtypes.to_dict()
                    for column_name in columns_name:
                        ProjectDatumColumns(
                            project_datum_id = pcr.project_data_id._id
                            , active = True
                            , name = column_name
                            , type = str(column_types[column_name])
                            , target = False
                            , updated_at = datetime.now()
                            , created_at = datetime.now()
                        ).save()
                    merged = pd.concat([df, df_1], axis=1)
                    data_buf = StringIO()
                    merged.to_csv(data_buf, index=False)
                    project_data.data = data_buf
                    project_data.save
                q.status = "completed"
                q.save()
            break


def main(db):
    pp = Preprocessor()
    pp.preprocess()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The preprocessor is a app that run sklearn machine learning algorithms one by one. A data is provided by the Orchestra.')

    parser.add_argument('-e', '--environment', help='specify environment', default='development')
    args = parser.parse_args()
    environment = args.environment
    main(setup_db(environment))
