import csv
import pandas as pd

file1 = "./diameter.csv"
file2 = "./sweep.csv"
csv_reader = pd.read_csv(open(file1), header=None)

print(csv_reader)

diameters = list(csv_reader.iloc[1])
print(diameters)

csv_reader = pd.read_csv(open(file2), header=None)
print(csv_reader)

for index in range(0, len(csv_reader.columns.tolist())):
    print(index)
    for row in range(0, 500):
        csv_reader.iloc[row, index] = float(csv_reader.iloc[row, index])/diameters[index]

csv_reader.to_csv("./new_sweep.csv")