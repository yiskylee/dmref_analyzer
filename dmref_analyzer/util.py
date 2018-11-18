import os
import re
import sys
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np

def skewness(x):
    n = len(x)
    x_mean = np.mean(x)
    x_centered = x - x_mean
    v = np.sum(np.square(x_centered)) / (n - 1)
    return np.max(x) / np.min(x), \
           np.sum(np.power(x_centered, 3)) / ((n - 1)*np.power(v, 1.5))

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def find_dir(dir_name, path):
    for root, dirs, files in os.walk(path):
        if dir_name in dirs:
            return os.path.join(root, dir_name)


def find_file_with_regex(base_dir, file_pattern, sample_id):
    file_paths = [os.path.abspath(os.path.join(root, file_name))
                  for root, dirs, files in os.walk(base_dir)
                  for file_name in files if re.match(file_pattern, file_name)]
    file_type = file_pattern.split("_")[0]
    if len(file_paths) == 1:
        return file_paths[0]
    elif len(file_paths) == 0:
        # print file_type + " file for sample " + str(sample_id) + " does not exist, skipping..."
        return None
    else:
        # if there are multiple data files, it must be R, RF, and F
        # F and R could have multiple data files.
        if file_type not in ['R', 'F', 'RF']:
            print "Multiple parameter / data files for sample " + \
                  str(sample_id) + " found: "
            for param_file in file_paths:
                print param_file
            raise Exception("exiting...")
        else:
            # there are duplicate names in multiple data files
            if len(set(file_paths)) != len(file_paths):
                print "Duplicated files found for sample " + \
                      str(sample_id)
                for file in file_paths:
                    print file
                raise Exception("exiting...")
            else:
                return file_paths

def find_files_with_regex(base_dir, file_pattern, sample_id):
    file_paths = [os.path.abspath(os.path.join(root, file_name))
                  for root, dirs, files in os.walk(base_dir)
                  for file_name in files if re.match(file_pattern, file_name)]
    file_type = file_pattern.split("_")[0]
    if len(file_paths) == 1:
        return file_paths
    elif len(file_paths) == 0:
        print file_type + " file for sample " + str(sample_id) + " does not exist, skipping..."
        return False
    else:
        if file_type != "F":
            print "Multiple parameter files for sample " + str(sample_id) + " found: "
            for param_file in file_paths:
                print param_file
            raise Exception("exiting...")

        else:
            # there are duplicate names in fusion data files
            if len(set(file_paths)) != len(file_paths):
                print "Duplicated files found for sample " + str(sample_id)
                for file in file_paths:
                    print file
                raise Exception("exiting...")
            else:
                return file_paths

def find_files_with_regex(base_dir, file_pattern):
    file_paths = [os.path.abspath(os.path.join(root, file_name))
                  for root, dirs, files in os.walk(base_dir)
                  for file_name in files if re.match(file_pattern, file_name)]
    return file_paths


def gen_total_fusion_time(fusion_file_paths):
    total_fusion_time = 0
    for fusion_file in fusion_file_paths:
        with open(fusion_file, 'rb') as fh:
            for line in fh:
                pass
            if ',' in line:
                fusion_time = float(line.split(',')[1])
            total_fusion_time += fusion_time
    return total_fusion_time


def gen_normalized_dataset(data, features, outcomes):
    if outcomes:
        return pd.DataFrame(
            data=scale(data[features+outcomes].as_matrix(), axis=0),
            columns=features + outcomes)
    else:
        return pd.DataFrame(
            data=scale(data[features].as_matrix(), axis=0),
            columns=features)

def scale_test_data(test_data, train_data):
    return (test_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)



def plot_fused_vs_orig(rf_mat, r_mat, feature):
    gs = gridspec.GridSpec(5, 2, width_ratios=[3, 1])
    fig = plt.figure(figsize=(10, 20))
    for i in np.arange(5):
        outcome = rf.outcomes[i]
        for j in np.arange(2):
            if j == 0:
                ax_rf = fig.add_subplot(gs[i, j])
                ax_rf.scatter(rf_mat[feature], rf_mat[outcome])
                ax_rf.set_xlabel(feature)
                ax_rf.set_ylabel(outcome)
            else:
                ax_r = fig.add_subplot(gs[i, j], sharey=ax_rf)
                ax_r.scatter([0] * np.size(r_mat[outcome]), r_mat[outcome])
                plt.setp(ax_r.get_yticklabels(), visible=False)
                plt.setp(ax_r.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig('CrossComparisonPlot' + feature + '.pdf')