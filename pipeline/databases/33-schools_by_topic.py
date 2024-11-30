#!/usr/bin/env python3
""" Retrieve documents by topic """


def schools_by_topic(mongo_collection, topic):
    """ Retrieves all documents matching a given topic

        - mongo_collection ... pymongo collection objet
        - topic .............. string, topic searched
    """
    return list(mongo_collection.find({"topics": {"$in": [topic]}}))
