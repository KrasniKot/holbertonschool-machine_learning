#!/usr/bin/env python3
""" Fetch the home planets of all sentient species """

import requests


def sentientPlanets():
    """ Returns the list of names of the home planets of all sentient species """
    # Fetch all species
    response = requests.get('https://swapi-api.hbtn.io/api/species').json()

    splanets = []
    while nurl := response.get('next'):  # Iterate over the pages
        species = response.get('results')
        for s in species:  # Iterate over the species
            if any('sentient' in (s.get(key) or '') for key in ('designation', 'classification')):
                if hworld := s.get('homeworld'):
                    splanets.append(requests.get(hworld).json().get('name'))

        response = requests.get(nurl).json()

    return splanets
