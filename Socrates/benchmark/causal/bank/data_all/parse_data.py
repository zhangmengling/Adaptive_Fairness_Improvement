import numpy as np
import ast
import os


input = open('./bank.csv', 'r')
lines = input.readlines()

print(len(lines))
y = []
all_data = []
i = 0
for line in lines:
    #array.append([int(x) for x in line.split(",")])

    array = [float(x) for x in line.split(",")]
    x = array[:(len(array) - 1)]

    all_data.append(x)

    output_x0 = open("data{}.txt".format(i), 'w+')
    output_x0.write("{}\n".format(x))
    output_x0.close()

    y.append(array[-1])

    i = i + 1

output_y = open('labels.txt', 'w+')
output_y.write("{}\n".format(y))
output_y.close()
input.close()
#
# data_all = "./"
#
# g_all = os.walk(data_all)
# train_data = []
# train_labels = []
# # with open(data_all_labels, "r") as f:
# #     all_labels = eval(f.read())
#
# exist_all_data = []
# files = os.listdir("./")
# num = len(files)
# print("-->num", num)
# for i in range(0, num):
#     file_name = "./" + "data" + str(i) + ".txt"
#     with open(file_name, "r") as f:
#         data = eval(f.readline())
#         exist_all_data.append(data)
#     try:
#         with open(file_name, "r") as f:
#             data = eval(f.readline())
#             exist_all_data.append(data)
#             # print(data)
#             # if data not in test_data:
#             #     train_data.append(data)
#             #     train_labels.append(all_labels[i])
#             # else:
#             #     print("in test_data")
#     except:
#         break
# print("-->all_data", all_data[:10])
# print("-->exist_all_data", exist_all_data[:10])