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
    updated_at = fields.DateTimeField()
    created_at = fields.DateTimeField()

    class Meta:
        final = True

class ProcessColumnsRequests(MongoModel):
    project_data_id = fields.ReferenceField(ProjectData)
    preprocess_algorithms_id = fields.ReferenceField(PreprocessAlgorithm)
    task = fields.CharField()
    target_columns = fields.ListField(fields.CharField())
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
    project_data_id = fields.ReferenceField(ProjectData)
    ml_algorithms_id = fields.ReferenceField(MachineLearningAlgorithms)
    task = fields.CharField()
    target_algorithms = fields.ListField(fields.CharField())
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


class ProjectDatumColumns(MongoModel):
    project_datum_id = fields.ReferenceField(ProjectData)
    active = fields.BooleanField()
    name = fields.CharField()
    type = fields.CharField()
    target = fields.BooleanField()
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

