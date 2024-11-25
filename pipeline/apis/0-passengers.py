#!/usr/bin/env python3
""" Fetch the number ships for a given number of passengers """

import requests


def availableShips(passengerCount):
    """ Fetches the number of available ships that can hold a given number of passengers """  # noqa

    def safe_convert_to_float(x):
        """ Function so safely convert to int without getting errors """
        # Replace commas with dots for decimal notation
        if ',' in x:
            x = x.replace(",", ".")

        # Validate if x is a valid float
        if x.replace(".", "", 1).isdigit():
            return float(x)

        return -1.0

    # Get response for a request
    response = requests.get('https://swapi-api.hbtn.io/api/starships/').json()

    aships = []  # List to hold available ships
    while nurl := response.get('next'):  # Loop over all pages
        ships = response.get('results')

        for ship in ships:  # Iterate over the ships
            if safe_convert_to_float(ship.get('passengers')) >= passengerCount:
                aships.append(ship.get('name'))

        response = requests.get(nurl).json()  # Go to next page

    return aships
