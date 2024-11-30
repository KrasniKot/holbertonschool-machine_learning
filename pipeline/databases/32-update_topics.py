
#!/usr/bin/env python3
""" Changes all topics of a collection based on their name """


def update_topics(mongo_collection, name, topics):
    """ Changes the topics of a collection document by the given name
        - mongo_collection ..... pymongo collection.
        - name ................. name of the document to update.
        - topics ............... topics to set for the school.
    """
    mongo_collection.update_many({"name": name},{"$set": {"topics": topics}})
