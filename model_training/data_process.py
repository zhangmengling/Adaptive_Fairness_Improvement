import csv
import numpy as np
import pandas as pd

file_path = "../preprocess_datasets/adult(numeric).csv"

def read_csv(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows

def numerate_features(rows):
    for i in range(1, len(rows)):
        asdf = 1

    with open('A.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row for row in reader]
    print(column)

def delete_empty(rows):
    new_rows = []
    for i in range(1, len(rows)):
        row = rows[i]
        empty_cell = " ?"
        if empty_cell not in row:
            new_rows.append(row)
    return new_rows

def write_csv(new_path, attribute_name, new_rows):
    writerCSV = pd.DataFrame(columns=attribute_name, data=new_rows)
    writerCSV.to_csv(new_path)

rows = read_csv(file_path)
attribute_names = rows[0]
print("-->attribute_names", attribute_names)
print("-->row29", rows[28])
new_rows = delete_empty(rows)
new_path = "../preprocess_datasets/adult(numeric)1.csv"
write_csv(new_path, attribute_names, new_rows)











