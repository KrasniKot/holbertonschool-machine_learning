#!/usr/bin/env python3
""" Log stats about Nginx logs stored in MongoDB """

from pymongo import MongoClient


def log_stats():
    """ Logs formatted stats about Nginx logs stored in MongoDB. """
    # ####### Connect to MongoDB
    client = MongoClient('mongodb://127.0.0.1:27017')  # Set client
    db = client.logs  # Set db
    col = db.nginx  # Set collection
    # #######

    # Number of documents in a collection
    print(f"{col.count_documents({})} logs")

    # Number of documents with each method
    print("Methods:")
    [print(f"\tmethod {m}: {col.count_documents({'method': m})}") for m in ["GET", "POST", "PUT", "PATCH", "DELETE"]]  # noqa

    # Number of documents of GET method with /status path
    print(f"{col.count_documents({"method": "GET", "path": "/status"})} status check")  # noqa


if __name__ == "__main__":
    log_stats()
