#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np

def getdata():
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

    for specie in species_codes['Specie Code']:
        if specie in benthic.keys():
            benthic.drop(columns=[specie],inplace=True)
    for label in ['FILE','FISH','NONE','PLGO','PLEW']:
        benthic.drop(columns=[label],inplace=True)       

    benthic['Herbivore (Grazer)'] =  biomass0['Herbivore (Grazer)']
    benthic['Herbivore (Scraper)'] =  biomass1['Herbivore (Scraper)']
    benthic['Herbivore (Browser)'] =  biomass2['Herbivore (Browser)']
    benthic['Detritivore (exclusively)'] =  biomass3['Detritivore (exclusively)']
    benthic['Corallivore'] =  biomass4['Corallivore']
    benthic['Planktivore'] =  biomass5['Planktivore']
    benthic['Small Invert. Feeder'] =  biomass6['Small Invert. Feeder']
    benthic['Large Invert. Feeder'] =  biomass7['Large Invert. Feeder']
    benthic['Small Predator'] =  biomass8['Small Predator']
    benthic['Large Predator'] =  biomass9['Large Predator']

    DistCoast = []
    DistStream = []
    Population = []
    Effluent = []
    UrbanIndex = []
    PointIndex = []
    AgrIndex = []
    FormPlIndex = []
    FragIndex = []
    DitchIndex = []

    sites = np.array(benthic['Site'])
    for i in range(0,len(sites)):
        if (len(sites[i])<7):
            tempsite = sites[i][0:4]+'0'+sites[i][4:]
            sites[i] = tempsite
    benthic['Site'] = sites

    human_sites = np.array(human['Site'])

    found = 0
    for site in benthic['Site']:
        idx = np.where(site==human_sites)[0]
        if len(idx)>0: 
            DistCoast.append(human['DistCoast'][idx[0]])
            DistStream.append(human['DistStream'][idx[0]])
            Population.append(human['Population'][idx[0]])
            Effluent.append(human['Effluent'][idx[0]])
            UrbanIndex.append(human['UrbanIndex'][idx[0]])
            PointIndex.append(human['PointIndex'][idx[0]])
            AgrIndex.append(human['AgrIndex'][idx[0]])
            FormPlIndex.append(human['FormPlIndex'][idx[0]])
            FragIndex.append(human['FragIndex'][idx[0]])
            DitchIndex.append(human['DitchIndex'][idx[0]])
        else:
            DistCoast.append(float(0))
            DistStream.append(float(0))
            Population.append(float(0))
            Effluent.append(float(0))
            UrbanIndex.append(float(0))
            PointIndex.append(float(0))
            AgrIndex.append(float(0))
            FormPlIndex.append(float(0))
            FragIndex.append(float(0))
            DitchIndex.append(float(0))

    DistCoast = np.array(DistCoast)
    DistStream = np.array(DistStream)
    Population = np.array(Population)
    Effluent = np.array(Effluent)
    UrbanIndex = np.array(UrbanIndex)
    PointIndex = np.array(PointIndex)
    AgrIndex = np.array(AgrIndex)
    FormPlIndex = np.array(FormPlIndex)
    FragIndex = np.array(FragIndex)
    DitchIndex = np.array(DitchIndex)

    benthic['DistCoast'] = DistCoast
    benthic['DistStream'] = DistStream
    benthic['Population'] = Population
    benthic['Effluent'] = Effluent
    benthic['UrbanIndex'] = UrbanIndex
    benthic['PointIndex'] = PointIndex
    benthic['AgrIndex'] = AgrIndex
    benthic['FormPlIndex'] = FormPlIndex
    benthic['FragIndex'] = FragIndex
    benthic['DitchIndex'] = DitchIndex

    benthic.to_csv('benthic_grouped.csv',index=False)

    return benthic, benthic_raw, human, species_codes, throphic_levels 