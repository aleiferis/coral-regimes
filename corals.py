#!/usr/bin/python3

import sys
import pandas as pd

benthic = pd.read_csv('./benthic.csv',skiprows=[0])
human = pd.read_csv('./human.csv')
species_codes = pd.read_csv('./species.csv')



