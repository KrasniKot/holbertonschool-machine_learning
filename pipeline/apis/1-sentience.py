#!/usr/bin/env python3
""" Fetch the home planets of all sentient species """

import requests


def sentientPlanets():
    """ Returns the list of names of the home planets of all sentient species """  # noqa
    # Fetch all species
    response = requests.get('https://swapi-api.hbtn.io/api/species').json()

    splanets = []
    while True:  # Keep fetching until the last page is reached
        species = response.get('results')
        for s in species:  # Iterate over the species
            if any('sentient' in (s.get(key) or '') for key in ('designation', 'classification')):  # noqa
                if hworld := s.get('homeworld'):
                    splanets.append(requests.get(hworld).json().get('name'))

        # Check if there is a next page
        nurl = response.get('next')
        if not nurl:
            break

        response = requests.get(nurl).json()  # Fetch the next page

    return splanets
