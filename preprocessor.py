# -*- coding: utf-8 -*-
from datetime import datetime
import abc
import argparse
from db import setup_db
from io import StringIO
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer
import pandas as pd
from models.models import ProcessColumnsRequestQueues, ProjectDatumColumns, ProcessColumnsRequestTargetColumns
from bson.objectid import ObjectId


class IPreprocess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def do(self):
        pass


class ImputerPreProcessor(IPreprocess):
    def do(self, data):
        imp = Imputer()
        transformed = imp.fit_transform(data.values.reshape(-1, 1))
        return imp, transformed

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
        elif type == 'Imputer':
            return ImputerPreProcessor()


class Preprocessor:
    def preprocess(self):
        while True:
            pcr_queues = ProcessColumnsRequestQueues.objects.raw({"status": "pendding"})
            for q in pcr_queues:
                pcr = q.process_columns_request_id
                project_data = pcr.project_datum_id
                data = project_data.data
                df = pd.read_csv(StringIO(data))
                if pcr.task == 'columns_type':
                    self.__process_column_type(df, pcr)
                    self.__describe(df, pcr)
                else:
                    tc = ProcessColumnsRequestTargetColumns.objects.raw({
                        'process_columns_request_id': pcr._id
                    })
                    columns = list(map(lambda x: x.project_datum_column_id, list(tc)))
                    p = PreprocessorFactory.create(pcr.task)
                    n = list(map(lambda x: x.name, columns))
                    p, processed_data = p.do(df.loc[:, n].dropna())
                    column_num = processed_data.shape[1] if len(processed_data.shape) > 1 else 1
                    columns_name = ["{0}.{1}".format(n, c) for c in range(0, column_num)]
                    df_1 = pd.DataFrame(processed_data, columns=columns_name)
                    column_types = df_1.dtypes.to_dict()
                    for column_name in columns_name:
                        ProjectDatumColumns(
                            project_datum_id = pcr.project_datum_id._id
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
                    project_data.data = data_buf.getvalue()
                    project_data.save()
                q.status = "completed"
                q.save()
            break

    def __process_column_type(self, df, pcr):
        types = df.dtypes.to_dict()
        for column_name, type in types.items():
            column = ProjectDatumColumns.objects.raw({
                'project_datum_id': pcr.project_datum_id._id,
                'name': column_name
            }).first()
            column.type = str(type)
            column.save()

    def __describe(self, df, pcr):
        desc = df.describe().to_dict()
        for column_name, v in desc.items():
            column = ProjectDatumColumns.objects.raw({
                'project_datum_id': pcr.project_datum_id._id,
                'name': column_name
            }).first()
            column.missing = len(df[column_name]) - df[column_name].count()
            column.mean = v['mean']
            column.std = v['std']
            column.count = v['count']
            column.quarter = v['25%']
            column.half = v['50%']
            column.three_quarters = v['75%']
            column.max = v['max']
            column.min = v['min']
            column.save()

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
