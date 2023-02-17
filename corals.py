#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import getdata

benthic, benthic_raw, human, species, throphic  = getdata.getdata()
print(benthic.keys())

plt.plot(benthic['Macroalgae'],benthic['Herbivore (Grazer)'],'o')
plt.grid()
plt.show()