import os
import pandas as pd
import numpy as np
from pyearth import Earth
from sklearn.preprocessing import scale
from dmref_analyzer.PlotGenerator import plot_learning_curve
from sklearn.model_selection import ShuffleSplit



def gen_v(df):
    voltages = df.v
    times = df.time
    v = voltages.max()

    loopLen = 4
    f = 1 / np.mean(np.diff(times.iloc[::loopLen]))

    even = np.arange(2, len(times) - 1, 2)
    odd = np.arange(1, len(times) - 1, 2)
    t = np.mean(times[even].as_matrix() - times[odd].as_matrix())
    T = np.sum(times[even].as_matrix() - times[odd].as_matrix())

    P = times.iloc[-1] - times.iloc[0]
    d = T / P
    return f, v, T, t, d


def gen_x(rng=np.arange(17, 10000)):
    paramFile = pd.read_csv('./parameter_rules.csv')
    sampleList = []
    fList = []
    vList = []
    TList = []
    tList = []
    dList = []
    gasAtmList = []
    for root, dirs, filenames in os.walk('../../Ji Hao/'):
        if len(filenames) > 1:
            for filename in filenames:
                sampleID = int(root.split('/')[-1])
                if sampleID in rng:
                    if 'F' in filename and 'IVF' not in filename:
                        inputFile = root + '/' + filename
                        df = pd.read_csv(inputFile)
                        del df['Unnamed: 4']
                        del df['Item']
                        df.columns = ['time', 'v', 'i']
                        (f, v, T, t, d) = gen_v(df)
                        sampleList.append(sampleID)
                        fList.append(f)
                        vList.append(v)
                        TList.append(T)
                        tList.append(t)
                        dList.append(d)
                    if filename == 'Parameters.txt':
                        inputFile = root + '/' + filename
                        paramMatrix = pd.read_csv(inputFile, delimiter='_', header=None)
                        paramMatrix.columns = paramFile.name.values
                        paramMatrix.index = paramMatrix.experimentType
                        #                         sampleID = int(paramMatrix.sampleID[0])
                        gasAtmList.append(paramMatrix.ix['F', 'gasAtmosphere'])

    df = pd.DataFrame({'sampleNum': sampleList,
                       'frequency': fList,
                       'voltage': vList,
                       'totalTimeVolOn': TList,
                       'timeVolOnPerPul': tList,
                       'dutyCycle': dList,
                       'gasAtmosphere': gasAtmList})
    df = df[['sampleNum', 'frequency', 'voltage', 'totalTimeVolOn', 'timeVolOnPerPul', 'dutyCycle', 'gasAtmosphere']]
    df.sort_values(by='sampleNum', inplace=True)
    df.set_index('sampleNum', inplace=True)
    return df


def gen_y(rng=np.arange(17, 10000)):
    return ((gen_r_after(rng) - gen_r_before(rng)) / gen_r_before(rng)).as_matrix().ravel()


def gen_r_after(rng=np.arange(17, 10000)):
    rList = []
    sampleList = []
    for root, dirs, filenames in os.walk('../../Ji Hao/'):
        for filename in filenames:
            if 'IVF' in filename:
                inputFile = root + '/' + filename
                df = pd.read_csv(inputFile)
                del df['Unnamed: 4']
                del df['Item']
                df.columns = ['v', 'i', 'r']
                rs = df.r
                r = np.mean(rs[rs != 0])
                sampleNum = int(filename.split('_')[1].split('.')[0])
                if sampleNum in rng:
                    sampleList.append(sampleNum)
                    rList.append(r)
    df = pd.DataFrame({'sampleNum': sampleList,
                       'resistance': rList})
    df = df[['sampleNum', 'resistance']]
    df.sort_values(by='sampleNum', inplace=True)
    df.set_index('sampleNum', inplace=True)
    return df


def gen_r_before(rng=np.arange(17, 10000)):
    rList = []
    sampleList = []
    for root, dirs, filenames in os.walk('../../Ji Hao/'):
        for filename in filenames:
            if 'IV' in filename and 'F' not in filename:
                sampleNum = int(root.split('/')[-1])
                if sampleNum in rng:
                    inputFile = root + '/' + filename
                    df = pd.read_csv(inputFile)
                    del df['Unnamed: 4']
                    del df['Item']
                    df.columns = ['v', 'i', 'r']
                    rs = df.r
                    r = np.mean(rs[rs != 0])
                    sampleList.append(sampleNum)
                    rList.append(r)
    df = pd.DataFrame({'sampleNum': sampleList,
                       'resistance': rList})
    df = df[['sampleNum', 'resistance']]
    df.sort_values(by='sampleNum', inplace=True)
    df.set_index('sampleNum', inplace=True)
    return df


def spline(X, y, features, outcome):
    # estimator = linear_model.Ridge(solver='svd')
    estimator = Earth(feature_importance_type='rss')
    feature_s = pd.Series(
        data=features, index=['x' + str(i) for i in np.arange(len(features))])
    table = []

    estimator.fit(X, y)
    # if estimator.rsq_ > 0.5:
    #     print outcome, 'r2 score: ', estimator.rsq_
    print '================='
    print 'Features Importance:'
    # printing feature importances with x0 to xn
    # mapped to real features name
    for line in estimator.summary_feature_importances().split('\n'):
        line_cleaned = line.split()
        if len(line_cleaned) == 2:
            print line_cleaned[0], line_cleaned[1], \
                feature_s[line_cleaned[0]]
    print 'r2 score: ', estimator.rsq_
    return estimator