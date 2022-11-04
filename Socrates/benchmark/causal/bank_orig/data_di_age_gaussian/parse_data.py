import numpy as np
import ast


input = open('data_3437.csv', 'r')
lines = input.readlines()



print(len(lines))
y = []
i = 0
for line in lines:
    #array.append([int(x) for x in line.split(",")])

    array = [float(x) for x in line.split(",")]
    x = array[:(len(array))]

    output_x0 = open("data{}.txt".format(i), 'w+')
    #output_x0.write(x)

    output_x0.write("{}\n".format(x))

    output_x0.close()

    #y.append(array[-1])

    i = i + 1

#output_y = open('labels.txt', 'w+')
#output_y.write("{}\n".format(y))

#output_y.close()
input.close()
