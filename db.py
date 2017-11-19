# -*- coding: utf-8 -*-
import yaml
from pymodm import connect, fields, MongoModel, EmbeddedMongoModel


def setup_db(environment):
    config = parse_db_config(environment)
    connect("mongodb://{0}/{1}".format(config['clients']['default']['hosts'][0], config['clients']['default']['database']))


def parse_db_config(environment):
    with open('config/mongodb.yml') as f:
        config = yaml.load(f)
    return config[environment]

