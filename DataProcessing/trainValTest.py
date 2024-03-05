#Philippe Joly
#MAIS 202

#This is to create training, validation, and testing splits for the different models
#The approach will be to store a range of indices associated with the sets for the specific methods

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

root=os.getenv("ROOT")
print("Working...")

df = pd.read_csv(root+'Data_2019-2022.csv')

df = df.dropna()
pow = df.pop('Average Power Output (MW)')
df['Average Power Output (MW)'] = pow

dfC = df.copy()

test = df.sample(n=2000)
df = df.drop(test.index)

df.to_csv(root+'train_cv_static.csv', index=False)

val = df.sample(n=2000)
df = df.drop(val.index)

df.to_csv(root+'train_static.csv', index=False)
val.to_csv(root+'val_static.csv', index=False)
test.to_csv(root+'test_static.csv', index=False)

#Time Series
test = dfC.tail(2000)
dfC = dfC.drop(test.index)

df.to_csv(root+'train_cv_time.csv', index=False)

val = dfC.tail(2000)
dfC = dfC.drop(val.index)

dfC.to_csv(root+'train_time.csv', index=False)
val.to_csv(root+'val_time.csv', index=False)
test.to_csv(root+'test_time.csv', index=False)

print("Done!")