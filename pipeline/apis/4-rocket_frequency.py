#!/usr/bin/env python3
""" Number of launches per rocket """

import requests
from collections import Counter


def get_launches():
    """ Fetches all the launches """
    # Fetch launches
    results = requests.get('https://api.spacexdata.com/v4/launches').json()

    # Get count of launches per rocket
    for k, v in Counter([lch.get('rocket') for lch in results]).most_common():
        rname = requests.get(f'https://api.spacexdata.com/v4/rockets/{k}').json().get('name')  # noqa
        print(f'{rname}: {v}')  # Print results


if __name__ == '__main__':
    get_launches()
