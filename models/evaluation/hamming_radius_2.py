import os
import sys
import numpy as np
import scipy.io as sio
import scipy as sp
import random
import cv2
import caffe
import os.path as opath
from multiprocessing import Pool

def hamming_radius2(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(validation_code, database_code.T)
    ground_truth = np.dot(validation_labels, database_labels.T)
    ground_truth[ground_truth>0] = 1.0 
    length_of_code = database_code.shape[1]
    radius2 = sim >= (length_of_code - 4)
    not_radius2 = sim < (length_of_code - 4)
    sim[radius2] = 1.0
    sim[not_radius2] = 0.0
    ground_truth[not_radius2] = 0.0
    count_radius2 = np.sum(sim, axis=1)
    for i in xrange(count_radius2.shape[0]):
        if count_radius2[i] == 0:
            count_radius2[i] = 1.0
    return np.mean(np.divide(np.sum(ground_truth, axis=1), count_radius2))

def load_code_and_label(params):
    path = params['path']
    params['database_code'] = np.load(opath.join(path, "database_code.npy"))
    params['database_labels'] = np.load(opath.join(path, "database_label.npy"))
    params['validation_code'] = np.load(opath.join(path, "validation_code.npy"))
    params['validation_labels'] = np.load(opath.join(path, "validation_label.npy"))

code_and_label = {}
code_and_label['path']= sys.argv[1]
load_code_and_label(code_and_label)
precision = hamming_radius2(code_and_label)
print precision
print "end"

