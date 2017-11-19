# -*- coding: utf-8 -*-
from datetime import datetime
import abc
import argparse
from db import setup_db
from io import StringIO
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from models.models import ProcessColumnsRequestQueues, ProjectDatumColumns


class IPreprocess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def do(self):
        pass


class OneHotEncoderPreProcessor(IPreprocess):
    def do(self, data):
        enc = OneHotEncoder()
        transformed = enc.fit_transform(data.reshape(-1, 1))
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
    def preprocess(self):
        while True:
            pcr_queues = ProcessColumnsRequestQueues.objects.all()
            for q in pcr_queues:
                pcr = q.process_columns_request_id
                project_data = pcr.project_data_id
                data = project_data.data
                df = pd.read_csv(StringIO(data))
                if pcr.task == '':
                    types = df.dtypes.to_dict()
                    for column_name, type in types.items():
                        column = ProjectDatumColumns.objects.raw({
                            'project_datum_id': pcr.project_data_id._id,
                            'name': column_name
                        }).first()
                        column.type = str(type)
                        column.save()
                else:
                    columns = ProjectDatumColumns.objects.raw({'project_datum_id': pcr.project_data_id._id }).values()
                    p = PreprocessorFactory.create(pcr.task)
                    p, processed_data = p.do(df.loc[:, list(map(lambda x: x['name'], columns))])
                    columns_name = ["{0}.{1}".format(columns, c) for c in range(0, len(processed_data.shape))]
                    df_1 = pd.DataFrame(processed_data, columns=columns_name)
                    column_types = df_1.dtypes.to_dict()
                    for column_name in column_name:
                        ProjectDatumColumns(
                            project_datum_id = pcr.project_data_id._id
                            , active = True
                            , name = column_name
                            , type = column_types[column_name]
                            , target = False
                            , update_at = datetime.now()
                            , created_at = datetime.now()
                        ).save()
                    merged = pd.concat([df, df_1], axis=1)
                    data_buf = StringIO()
                    merged.to_csv(data_buf, index=False)
                    project_data.data = data_buf
                    project_data.save
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
