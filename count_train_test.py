import os
import numpy as np
directory = 'data/users/'
train_num = []
test_num = []
tot_num = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename)) as f:
           last_line = f.readlines()[-1]
           train_num.append(int(last_line.strip().split(", ")[0].split(": ")[1]))
           test_num.append(int(last_line.strip().split(", ")[1].split(": ")[1]))
           tot_num.append(int(last_line.strip().split(", ")[0].split(": ")[1]) + int(last_line.strip().split(", ")[1].split(": ")[1]))
    else:
        continue

print(train_num)
print(test_num)

print(len(train_num))
print(len(test_num))

print(np.mean(train_num))
print(np.mean(test_num))
print(np.mean(tot_num))
print(max(set(tot_num), key=tot_num.count))
print(np.median(tot_num))

print(sorted(tot_num))
