# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:32:02 2021

@author: T530
"""

import os
import re
import pickle
metrics = ["Triplet MRR", "Subset MRR"]
directory = r'/home/mb88814/manifold_alignment/output/users'
users = dict()
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename+"/results.txt"), "r") as f:
        users[filename] = dict()
        for i, line in enumerate(f.readlines()):
            users[filename][metrics[i]] = re.findall("\d+\.\d+", line)[0]
            
pickle.dump(users, open( "/home/mb88814/manifold_alignment/plots/users_results.pkl", "wb" ) )