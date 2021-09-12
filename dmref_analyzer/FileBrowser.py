from _typeshed import StrOrBytesPath
import glob
import os
import fnmatch
import re
import numpy as np
import util

import pandas as pd
import sys
from os.path import dirname, abspath


class FileBrowser(object):
    def __init__(self, root_dir=None, param_rule_file=None):
        if root_dir is None:
            self.root_dir = dirname(dirname(abspath(__file__)))
        else:
            self.root_dir = root_dir
        if param_rule_file is None:
            param_rule_file = os.path.expanduser(
                './parameter_rules.csv')
        self.param_rule = pd.read_csv(param_rule_file, index_col=1)

    def show_file_with_ext(self, file_ext):
        files_with_ext = [os.path.join(root, f)
                          for root, dirs, files in os.walk(self.root_dir)
                          for f in fnmatch.filter(files, '*.' + file_ext)]
        for file in files_with_ext:
            print(file)

    def walk_dir(self):
        for (root, dirs, filenames) in os.walk(self.root_dir):
            print("ROOT: ", root)
            print("DIRS: ", dirs)
            print("FIL)ES: ", filenames)
            print("================================================")

    def show_param(self, sample_id):
        param_file_pattern = "Parameters_" + str(sample_id) + "(_\d)*" + ".csv"
        param_file = util.find_file_with_regex(
            self.root_dir, param_file_pattern, sample_id)
        return pd.read_csv(param_file)

    def show_sample_with_experiments(self, sample_rng=np.arange(1, 10000)):
        all_experiment_types = list(map(
            str.strip, self.param_rule.loc['experimentType', 'options'].split(',')))
        df_columns = all_experiment_types
        # Used to count number of fusion files as well
        # + ['numFusionFiles'] + ['totalFusionTime']
        # Pre-allocate rows, the result usually contains less number of rows
        file_summary_df = pd.DataFrame(index=sample_rng, columns=df_columns)
        file_summary_df.index.name = 'sampleID'
        for sample_id in sample_rng:
            param_file_pattern = "Parameters_" + \
                str(sample_id) + "(_\d)*" + ".csv"
            # fusion_file_pattern = "F_" + str(sample_id) + "(_\d)*" + ".csv"
            param_file = util.find_file_with_regex(
                self.root_dir, param_file_pattern, sample_id)
            # fusion_file_paths = util.find_files_with_regex(input_dir, fusion_file_pattern)
            if not param_file:
                continue
            # if fusion_file_paths:
            #     num_fusion_files = len(fusion_file_paths)
            #     total_fusion_time = util.gen_total_fusion_time(fusion_file_paths)
            #     file_summary_df.loc[row_num]["numFusionFiles"] = num_fusion_files
            #     file_summary_df.loc[row_num]["totalFusionTime"] = total_fusion_time
            param_df = pd.read_csv(param_file)
            experiment_types = list(map(str.strip, param_df['experimentType']))
            entry = [
                'Yes' if exp in experiment_types else 'No' for exp in all_experiment_types]
            entry_series = pd.Series(index=all_experiment_types, data=entry)
            # Use loc or ix to append rows to existing data frames
            # iloc doesn't work in this case
            file_summary_df.loc[sample_id] = entry_series
            # file_summary_df.loc[row_num]['sampleID'] = sample_id
        file_summary_df.dropna(axis=0, how='all', inplace=True)
        # file_summary_df = file_summary_df.ix[:, (file_summary_df != 0).any(axis=0)]
        # file_summary_df.sort_values(by='sampleID', inplace=True)
        # file_summary_df.set_index('sampleID', inplace=True)
        return file_summary_df
