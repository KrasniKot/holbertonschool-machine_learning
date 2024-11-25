#!/usr/bin/env python3
""" Get location of a specific GitHub user """

import requests
from sys import argv
import time


def success(result):
    """ In case of successful request """
    print(result.json()['location'])


def nfound(result=None):
    """ In case user not found """
    print('Not found')


def lreached(result):
    """ In case rate limit reached """
    rlim    = int(result.headers['X-Ratelimit-Reset'])
    minutes = int((rlim - int(time.time())) / 60)

    print(f'Reset in {minutes} min')


if __name__ == '__main__':

    # Set possible responses
    responses = {
        200: success,
        403: lreached,
        404: nfound
    }

    # Request user
    got = requests.get(argv[1], headers={'accept': 'application/vnd.github+json'})

    # Handle response by code
    responses[got.status_code](got)
