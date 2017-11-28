# -*- coding: utf-8 -*-
from pymodm import fields, MongoModel

class MachineLearningAlgorithms(MongoModel):
    category = fields.CharField()
    module_name = fields.CharField()
    class_name = fields.CharField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True

class PreprocessAlgorithm(MongoModel):
    category = fields.CharField()
    module_name = fields.CharField()
    class_name = fields.CharField()
    # updated_at = fields.DateTimeField()
    # created_at = fields.DateTimeField()

    class Meta:
        final = True

class Projects(MongoModel):
    name = fields.CharField()
    file_name = fields.CharField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True

class ProjectData(MongoModel):
    project_id = fields.ReferenceField(Projects)
    data = fields.CharField()
    name = fields.CharField()
    description= fields.CharField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True


class ProjectDatumColumns(MongoModel):
    project_datum_id = fields.ReferenceField(ProjectData)
    active = fields.BooleanField()
    name = fields.CharField()
    type = fields.CharField()
    missing = fields.IntegerField()
    count = fields.IntegerField()
    mean = fields.FloatField()
    std = fields.FloatField()
    min = fields.FloatField()
    quarter = fields.FloatField()
    half = fields.FloatField()
    three_quarters = fields.FloatField()
    max = fields.FloatField()
    target = fields.BooleanField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True


class ProcessColumnsRequests(MongoModel):
    project_datum_id = fields.ReferenceField(ProjectData)
    preprocess_algorithm_id = fields.ReferenceField(PreprocessAlgorithm)
    task = fields.CharField()
    target_columns = fields.ListField(fields.CharField())
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True


class ProcessColumnsRequestTargetColumns(MongoModel):
    process_columns_request_id = fields.ReferenceField(ProjectData)
    project_datum_column_id = fields.ReferenceField(ProjectDatumColumns)
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True


class ProcessColumnsRequestQueues(MongoModel):
    process_columns_request_id = fields.ReferenceField(ProcessColumnsRequests)
    status = fields.CharField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True


class TrainingRequests(MongoModel):
    project_datum_id = fields.ReferenceField(ProjectData)
    machine_learning_algorithm_id = fields.ReferenceField(MachineLearningAlgorithms)
    task = fields.CharField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True


class TrainingRequestQueues(MongoModel):
    training_request_id = fields.ReferenceField(TrainingRequests)
    status = fields.CharField()
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True



class ClassificationTrainingResults(MongoModel):
    training_request_id = fields.ReferenceField(TrainingRequests)
    accuracy = fields.CharField()
    recall = fields.CharField()
    f1 = fields.CharField()

    class Meta:
        final = True

class RegressionTrainingResults(MongoModel):
    training_request_id = fields.ReferenceField(TrainingRequests)
    mse = fields.CharField()
    mae = fields.CharField()
    r2 = fields.CharField()

    class Meta:
        final = True

