import scipy.interpolate as inter
import matplotlib.gridspec as gridspec
import dmref_analyzer.ModelSelection as ms
import dmref_analyzer.FeatureSelection as fs
import dmref_analyzer.FileBrowser as fb
from dmref_analyzer import DataMatrix
import dmref_analyzer.util as util
import dmref_analyzer.PlotGenerator as pg
import dmref_analyzer.RegressionModel as rm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
from os.path import dirname, abspath
import os
import pandas as pd
import matplotlib

matplotlib.use("agg")


matplotlib.rc("font", size=20)


def baseLineCorrection(y):
    import peakutils

    baseline = peakutils.baseline(y)
    return y - baseline


def preprocess(df, peak_pos, align=True):
    # resample, normalize, and shift to ref peak position
    ref_peak = 1342.52
    wavelength = df.iloc[:, 0]
    intensity = df.iloc[:, 1]
    if align:
        wavelength += ref_peak - peak_pos
    interpo = inter.interp1d(wavelength, intensity)
    wavelength = np.arange(100, 3500)
    intensity = interpo(wavelength)
    intensity = intensity / max(intensity)
    return wavelength, intensity


def gen_all_df(sample_rng, align=True):
    base_dir = os.join(dirname(dirname(abspath(__file__))),  "RamanData")
    df_list = []
    for sample_id in sample_rng:
        data_dir_path = os.path.join(base_dir, str(sample_id))
        outcome_name = "Outcomes_" + str(sample_id) + ".csv"
        outcome_file_path = os.path.join(data_dir_path, outcome_name)
        outcome_df = pd.read_csv(outcome_file_path)
        for root, dirs, filenames in os.walk(data_dir_path):
            for file_name in filenames:
                if ("R" in file_name or "RF" in file_name) and not file_name.startswith(
                    "."
                ):
                    pos = file_name.split("_")[-1].split(".")[0]
                    data_file_path = os.path.join(root, file_name)
                    df = pd.read_csv(data_file_path, header=None)
                    g_peak_pos = outcome_df["gPeakPosition"][int(pos) - 1]
                    wavelength, intensity = preprocess(df, g_peak_pos, align)
                    intensity = baseLineCorrection(intensity)
                    name = str(sample_id) + "_" + pos
                    df1 = pd.DataFrame(data={name: intensity})
                    df_list.append(df1)
    return df_list


rf = DataMatrix(experiment_type="RF", sample_rng=np.arange(305, 363))
r = DataMatrix(experiment_type="R", sample_rng=np.arange(300, 305))

rmat = r.cleaned_mat()
rfmat = rf.cleaned_mat()

grp = rfmat.groupby(["voltage", "freq", "totalNumOfCycles"])

sampleToCond = {}
for key, item in grp:
    conditions = []
    for condition in key:
        if condition.is_integer():
            conditions.append(str(int(condition)))
        else:
            conditions.append("{:.1f}".format(condition))

    for sampleID in set(grp.get_group(key).sampleID):
        sampleToCond[str(sampleID)] = "_".join(conditions)

for sampleID in set(rmat.sampleID):
    sampleToCond[str(sampleID)] = "Original"

# Some modification
for sampleID in np.arange(315, 320):
    sampleToCond[str(sampleID)] = "9_2.5_2000"
for sampleID in np.arange(330, 335):
    sampleToCond[str(sampleID)] = "9_2.5_2000(2)"

all_df = pd.concat(gen_all_df(np.arange(300, 363)), axis=1)
all_df = all_df.transpose()
all_df = all_df.reset_index()
all_df["index"] = [sampleToCond[ind.split("_")[0]] for ind in all_df["index"]]
mean_df = all_df.groupby("index").mean()
mean_df1 = mean_df.drop(mean_df.index[5])


pg.plot_dendro(mean_df1.as_matrix(), mean_df1.index, metric="cityblock")
