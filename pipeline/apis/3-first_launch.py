#!/usr/bin/env python3
""" Get first upcomming launch """

import requests


def get_upcoming():
    """ Get first upcoming launch """
    ul = requests.get('https://api.spacexdata.com/v4/launches/upcoming').json()

    ul = min(ul, key=lambda launch: launch['date_unix'])

    rn = requests.get(f"https://api.spacexdata.com/v4/rockets/{ul.get('rocket')}").json().get('name')  # noqa
    li = requests.get(f"https://api.spacexdata.com/v4/launchpads/{ul.get('launchpad')}").json()  # noqa
    ln = li.get('name')
    ll = li.get('locality')

    print(f"{ul.get('name')} ({ul.get('date_local')}) {rn} - {ln} ({ll})")


if __name__ == '__main__':
    get_upcoming()
