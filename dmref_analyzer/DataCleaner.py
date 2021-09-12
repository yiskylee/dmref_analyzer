import util
import pandas as pd
import sys
import re
from os.path import dirname, abspath


def rename_param(input_dir, old_name, new_name):
    param_pattern = "Parameters_(\d\d*).csv"
    param_files = util.find_files_with_regex(input_dir, param_pattern)
    for param_file in param_files:
        param_df = pd.read_csv(param_file)
        if old_name not in param_df.columns:
            print("Old name does not exist in " + param_file + " continue...")
            continue
        else:
            param_df.rename(columns={old_name: new_name}, inplace=True)
            # sample_id = re.match(param_pattern, param_file.split('/')[-1]).group(1)
            # if int(sample_id) == 528:
            #     param_df.to_csv("./528.csv", index=False)
            param_df.to_csv(param_file, index=False)


def update_param_file(input_dir, new_param_rule_file):
    param_pattern = "Parameters_(\d\d*).csv"
    param_files = util.find_files_with_regex(input_dir, param_pattern)
    new_params = pd.read_csv(new_param_rule_file)['name']
    new_param_empty_df = pd.DataFrame(columns=new_params)
    for param_file in param_files:
        param_df = pd.read_csv(param_file)
        sample_id = re.match(param_pattern, param_file.split('/')[-1]).group(1)
        new_param_df = pd.merge(param_df, new_param_empty_df, how='outer')
        # Merge does not preserve the column order
        # Rearrange the column order to match that of new_param_empty_df
        new_param_df = new_param_df[new_params]
        new_param_df.to_csv(param_file, index=False)


if __name__ == "__main__":
    path = dirname(dirname(abspath(__file__)))
    new_param_rule_file = "./parameter_rules_new.csv"
    update_param_file(path, new_param_rule_file)
    # rename_param(path, "twistOrientation", "twistOrientation")
