#! /bin/python3
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import numpy as np

# found the prediction become whether the value is or not a?
X = pd.read_csv('test.csv')
pLetters = X['Prediction']

Y = pd.read_csv('aSubmission.csv')
