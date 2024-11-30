#!/usr/bin/env python3
""" Get a document insterted in a mongo collection """


def insert_school(mongo_collection, **kwargs):
    """ Inserts a new document in a mongo collection
        - mongo_collection ... pymongo collection
        - kwargs ............. document parameters
    """
    return mongo_collection.insert_one(kwargs).inserted_id