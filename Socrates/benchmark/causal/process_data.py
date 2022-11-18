import csv
import os

file_path = "bank/"

data_all = file_path + "data_all/"
data_all_labels = file_path + "data_all/labels.txt"
data_test = file_path + "data_test/"
data_train = file_path + "data_train/"

g_test = os.walk(data_test)
test_data = []
for path, dir_list, file_list in g_test:
    for file_name in file_list:
        # print(os.path.join(path, file_name))
        with open(os.path.join(path, file_name), "r") as f:
            # print(eval(f.read()))
            test_data.append(eval(f.read()))

# print("-->test_data:", test_data)

g_all = os.walk(data_all)
train_data = []
train_labels = []
with open(data_all_labels, "r") as f:
    all_labels = eval(f.read())

files = os.listdir(data_all)
num = len(files)
for i in range(0, num):
    file_name = data_all + "data" + str(i) + ".txt"
    try:
        with open(file_name, "r") as f:
            data = eval(f.readline())
            # print(data)
            if data not in test_data:
                train_data.append(data)
                train_labels.append(all_labels[i])
            # else:
            #     print("in test_data")
    except:
        break

print("length of train_data", len(train_data))
print("length of test_data", len(test_data))
print("length of data_all", num)

print("-->test_data", test_data[:10])
print("-->train_data", train_data[:10])

if not os.path.exists(data_train):
    os.makedirs(data_train)

for i in range(0, len(train_data)):
    x = train_data[i]
    output_x0 = open(data_train + "data{}.txt".format(i), 'w+')
    output_x0.write("{}\n".format(x))
    output_x0.close()

output_y = open(data_train + 'labels.txt', 'w+')
output_y.write("{}\n".format(train_labels))