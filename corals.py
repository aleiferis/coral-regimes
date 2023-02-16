#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np

benthic = pd.read_csv('./benthic.csv',skiprows=[0])
human = pd.read_csv('./human.csv')
species_codes = pd.read_csv('./species.csv')
throphic_levels = pd.read_csv('./throphic.csv')
benthic_raw = benthic

species0 = species_codes[species_codes['Trophic level'] == 'Herbivore (Grazer)']
species1 = species_codes[species_codes['Trophic level'] == 'Herbivore (Scraper)']
species2 = species_codes[species_codes['Trophic level'] == 'Herbivore (Browser)']
species3 = species_codes[species_codes['Trophic level'] == 'Detritivore (exclusively)']
species4 = species_codes[species_codes['Trophic level'] == 'Corallivore']
species5 = species_codes[species_codes['Trophic level'] == 'Planktivore']
species6 = species_codes[species_codes['Trophic level'] == 'Small Invert. Feeder']
species7 = species_codes[species_codes['Trophic level'] == 'Large Invert. Feeder']
species8 = species_codes[species_codes['Trophic level'] == 'Small Predator']
species9 = species_codes[species_codes['Trophic level'] == 'Large Predator']


print(species_codes['Trophic level'].unique())
print(len(species_codes['Trophic level']))
unused = []

biomass0 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Herbivore (Grazer)'])
for specie in species0['Specie Code']:
    if (specie in benthic.keys()):
        biomass0['Herbivore (Grazer)'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass1 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Herbivore (Scraper)'])
for specie in species1['Specie Code']:
    if (specie in benthic.keys()):
        biomass1['Herbivore (Scraper)'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass2 = pd.DataFrame(np.zeros( len(benthic['Site'])), columns=['Herbivore (Browser)'])
for specie in species2['Specie Code']:
    if (specie in benthic.keys()):
        biomass2['Herbivore (Browser)'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass3 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Detritivore (exclusively)'])
for specie in species3['Specie Code']:
    if (specie in benthic.keys()):
        biomass3['Detritivore (exclusively)'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass4 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Corallivore'])
for specie in species4['Specie Code']:
    if (specie in benthic.keys()):
        biomass4['Corallivore'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass5 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Planktivore'])
for specie in species5['Specie Code']:
    if (specie in benthic.keys()):
        biomass5['Planktivore'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass6 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Small Invert. Feeder'])
for specie in species6['Specie Code']:
    if (specie in benthic.keys()):
        biomass6['Small Invert. Feeder'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass7 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Large Invert. Feeder'])
for specie in species7['Specie Code']:
    if (specie in benthic.keys()):
        biomass7['Large Invert. Feeder'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass8 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Small Predator'])
for specie in species8['Specie Code']:
    if (specie in benthic.keys()):
        biomass8['Small Predator'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

biomass9 = pd.DataFrame(np.zeros(len(benthic['Site'])), columns=['Large Predator'])
for specie in species9['Specie Code']:
    if (specie in benthic.keys()):
        biomass9['Large Predator'] += benthic[specie].astype("float")
    else:
        unused.append(specie)

print(benthic.shape)
for specie in species_codes['Specie Code']:
    if specie in benthic.keys():
        benthic.drop(columns=[specie],inplace=True)
print(benthic.shape)

benthic['Herbivore (Grazer)'] =  biomass0['Herbivore (Grazer)']
print(benthic.keys())

print(unused)
