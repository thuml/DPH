import os
import sys
import numpy as np
import scipy.io as sio
import scipy as sp
import random
import cv2
import caffe
from multiprocessing import Pool
import os.path as opath
import scipy.io as ioc

def top_k(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(validation_code, database_code.T)
    ranking = np.argsort(-sim, axis=1)
    ground_truth = np.dot(validation_labels, database_labels.T)
    ground_truth[ground_truth>0] = 1.0
    top_k_result = []
    for i in xrange(10):
        ranking_k = ranking < params["k"]
        not_ranking_k = ranking >= params["k"]
        ground_truth[not_ranking_k] = 0.0
        top_k_result.append(np.mean(np.sum(ground_truth, axis=1)/float(params["k"])))
        params["k"] -= 100
    top_k_result.append(top_k_result[-1])
    top_k_result.reverse()
    return np.array(top_k_result)
        
def load_code_and_label(params):
    path = params['path']
    params['database_code'] = np.load(opath.join(path, "database_code.npy"))
    params['database_labels'] = np.load(opath.join(path, "database_label.npy"))
    params['validation_code'] = np.load(opath.join(path, "validation_code.npy"))
    params['validation_labels'] = np.load(opath.join(path, "validation_label.npy"))

code_and_label = {}
code_and_label['path']= sys.argv[1]
load_code_and_label(code_and_label)
code_and_label['k'] = 1000
result = top_k(code_and_label)
os.system("mkdir -p "+sys.argv[2])
ioc.savemat(opath.join(sys.argv[2], "top_k.mat"), {"x":np.array(range(0, 1001, 100)), "y":result})
print result
print "end"
