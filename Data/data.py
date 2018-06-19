import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

cwd = os.getcwd()
sys.path.append(cwd+'~/PINES/')

PINES = pd.read_csv("../S1_Data.csv")

nibFilename = np.array(PINES['Files'])
#image = np.zeros_like(nibFilename)
image = np.zeros_like(nibFilename)
image1 = np.zeros_like(nibFilename)

for p in range(len(nibFilename)):
	image[p] = nib.load(nibFilename[p])
	image1[p] = image[p].get_data()

print(image1[0])
np.save('../X.npy', image1)
