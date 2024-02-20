#Philippe Joly
#MAIS 202

#This is to create training, validation, and testing splits for the different models
#The approach will be to store a range of indices associated with the sets for the specific methods

import pandas as pd
root='C:\\Users\\philj\\OneDrive - McGill University\\COMP\\MAIS\\PROJECT\\data\\'

print("Working...")

df = pd.read_csv(root+'Data_2019-2022.csv')

# print(f'The length of the data set is', df.shape[0])
# t = input("Choose Test Set Length:")
# v = input("Choose validation Set Lenght:")

# if (t+v>= df.shape[0]):
#     raise Exception("incorrect input")


#Static (SVR, DNN)
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