import sys
import scipy
import numpy
import matplotlib
import pandas
import matplotlib.pyplot as plt
import numpy as np
from cmath import sqrt

# import data and assign feature names

data = "folder_path/Volatility scaling_python_test_2.csv"
names = ["Date", "Neo Industrial", "Herantis Pharma", "Componenta", "Zeeland Family", "Aspocomp", "Finnair", "Cleantech Invest", "Talenom", "QPR Software", "Soprano", "Etteplan", "Honkarakenne", "Nixu", "Poyry"]
dataset = pandas.read_csv(data, sep=";", names=names)

# preview data

print(dataset.shape)
print(dataset.head(20))
print(dataset.dtypes)

#Get rid of "date" column so as to not interfere with matrix multiplication
del dataset["Date"]

#Confirm this worked
# If you change the code, you may want to check calculations by entering this:print(dataset.shape)

# Calculate the weights to be multiplied with
weightt = 1/(len(dataset.columns))
# If you change the code, you may want to check calculations by entering this:print(round(weightt, 5))

#Create the array with the weights
weightt_vector = pandas.Series(round(weightt, 5), index=dataset.index)
# If you change the code, you may want to check calculations by entering this:print(weightt_vector)

#Make the arrays numpy arrays for matrix multiplication
dataset_numpy = np.array(dataset)
weightt_vector_numpy = np.array(weightt_vector)
#Testing print here to make sure nothing has gone wrong in changing arrays to numpy arrays
# If you change the code, you may want to check calculations by entering this:print(dataset_numpy)

#Resize for multiplication
weightt_vector_numpy.resize(len(dataset.index), 1)
# If you change the code, you may want to check calculations by entering this:print(weightt_vector_numpy)

#Multiply returns with appropriate weights
weighted_returns = dataset_numpy*weightt_vector_numpy
# If you change the code, you may want to check calculations by entering this:print(weighted_returns)

# Add the weighted returns to form portfolio returns
weighted_returns_sum = numpy.sum(weighted_returns, axis=1)
# If you change the code, you may want to check calculations by entering this: print(weighted_returns_sum)

# Resize to get a column vector
weighted_returns_sum.resize(len(dataset.index), 1)
# If you change the code, you may want to check calculations by entering this:print(weighted_returns_sum)

#Now raise the weighted_returns_sum to the power of two i.e. square it
weighted_returns_sum_squared = np.square(weighted_returns_sum)
# If you change the code, you may want to check calculations by entering this:print(weighted_returns_sum_squared)

#Now sum the squares to get realized variance of last 6 months
realized_var = numpy.sum(weighted_returns_sum_squared, axis=0)
# If you change the code, you may want to check calculations by entering this:print(realized_var)

#Now define trading days per month, per 6 months and annualized target volatility as well as monthly target volatility
trading_days_6_mo = 126
trading_days_1_mo = 21
annual_target_vola = 0.12
monthly_target_vola = annual_target_vola/sqrt(12)
# If you change the code, you may want to check calculations by entering this:print(monthly_target_vola)

#Multiply summed squares with monthly trading days and divide with 6 month trading days
realized_var_multidiv = realized_var * trading_days_1_mo / trading_days_6_mo
# If you change the code, you may want to check calculations by entering this: print(realized_var_multidiv)

#Take the square root of the former
estimated_vola = sqrt(realized_var_multidiv)
print("\n Estimated volatility: {:.2f} ".format(estimated_vola))

#Divide monthly target vola with estimated vola to get the weight for RMIM
rmim_weight = monthly_target_vola/estimated_vola
print("\n RMIM weight for the next month: {:.2f} ".format(rmim_weight))

