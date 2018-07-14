import os
import sys
import numpy as np
import scipy.io as sio
import scipy as sp
import random
import cv2
import caffe
from multiprocessing import Pool
import scipy.io as ioc
import os.path as opath

def get_precision(i):
    print i
    code_length1 = code_length
    precision = []
    recall = []
    label = validation_labels[i, :]
    label[label == 0] = -1
    all_sim_num = np.sum(ground_truth[:, i])
    for j in xrange(database_code.shape[1]):
      sim_num_l = np.sum(sim[:, i] >= code_length1)
      idx = ids[:, i]
      relevant_num = np.sum(ground_truth[idx[0:int(sim_num_l)], i])
      if relevant_num != 0:
          precision.append(relevant_num / float(sim_num_l))
          recall.append(relevant_num / float(all_sim_num))
      else:
          precision.append(0)
          recall.append(0)
      code_length1 -= 2
    if all_sim_num > 0:
      precision.append(precision[-1])
    else:
      precision.append(0.)
    recall.append(1)
    y = np.interp(query_interp_points, recall, precision)
    return y

def load_code_and_label(params):
    path = params['path']
    params['database_code'] = np.load(opath.join(path, "database_code.npy"))
    params['database_labels'] = np.load(opath.join(path, "database_label.npy"))
    params['validation_code'] = np.load(opath.join(path, "validation_code.npy"))
    params['validation_labels'] = np.load(opath.join(path, "validation_label.npy"))

code_and_label = {}
code_and_label['path']= sys.argv[1]
load_code_and_label(code_and_label)

database_code = np.array(code_and_label['database_code'])
validation_code = np.array(code_and_label['validation_code'])
database_labels = np.array(code_and_label['database_labels'])
validation_labels = np.array(code_and_label['validation_labels'])
query_num = validation_code.shape[0]
    
database_code = np.sign(database_code)
validation_code = np.sign(validation_code)

sim = np.dot(database_code, validation_code.T)
ids = np.argsort(-sim, axis=0)
code_length = database_code.shape[1]
query_interp_points = np.array(range(101)) / 100.0
ground_truth = np.dot(database_labels, validation_labels.T)
ground_truth[ground_truth > 0] = 1.
all_p = []
p = Pool(8)
result = p.map(get_precision, range(query_num))
for y in result:
  all_p.append(y)
P = np.mean(np.array(all_p), axis=0)
recall, precision = [query_interp_points, P]
os.system("mkdir -p "+sys.argv[2])
ioc.savemat(opath.join(sys.argv[2], "pr_result.mat"), {"recall":recall, "precision":precision})
print precision, recall
print "end"
