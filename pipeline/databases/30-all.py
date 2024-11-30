
#!/usr/bin/env python3
""" Retrieve all documents in a mongo  collection """


def list_all(mongo_collection):
    """ Lists all documents in a MongoDB collection.
        - mongo_collection .... the pymongo collection object.
    """
    return list(mongo_collection.find())
