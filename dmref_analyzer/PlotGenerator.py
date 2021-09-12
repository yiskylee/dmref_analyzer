import sys
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import scale
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def plot_dendro(X, labels, metric='euclidean'):
    # labels must be np array, otherwise indexing in dendrogram is problematic
    labels = np.array(labels)
    Y = linkage(pdist(X, metric=metric), method='ward')
    fig = plt.figure(figsize=(15, 15))
    ax_heat = fig.add_axes([0, 0, 1, 0.7])
    ax_dendro = fig.add_axes([0, 0.80, 1, 0.2])
    Z = dendrogram(Y, ax=ax_dendro, labels=labels, leaf_font_size=30,
                   leaf_rotation=30)
    ax_dendro.yaxis.set_visible(False)
    for pos in ['left', 'right', 'bottom', 'top']:
        ax_dendro.spines[pos].set_visible(False)
    index = Z['leaves']
    dist_mat = pairwise_distances(X, metric=metric)
    dist_mat = dist_mat[:, index]
    dist_mat = dist_mat[index, :]
    # ax_heat.matshow(X, aspect='auto', cmap=plt.cm.Reds)
    cax = ax_heat.matshow(dist_mat, aspect='auto', cmap=plt.cm.jet_r)
    # cax = ax_heat.matshow(dist_mat, aspect='auto', cmap=plt.cm.gray)
    ax_heat.get_xaxis().set_visible(False)
    # ax_heat.axis('off')
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels[index], fontsize=30)
    # plt.rc('font', size=20)
    plt.savefig('./tests/figures/dendro.png', dpi=600, bbox_inches='tight')
    plt.savefig('./tests/figures/dendro.pdf', bbox_inches='tight')


def plot_outcome_vs_feature(feature, outcome, df, plot_type='scatter'):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel(feature)
    ax.set_ylabel(outcome)

    if plot_type == 'scatter':
        ax.scatter(df[feature], df[outcome])
    elif plot_type == 'line':
        ax.plot(df[feature], df[outcome])

    ax.set_title(outcome + ' vs ' + feature)


def plot_outcome_vs_feature_ten_vs_tenf(feature, outcome, df,
                                        plot_type='scatter'):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(feature)
    ax.set_ylabel(outcome)
    ten = df[df['experimentType'] == 'Ten']
    tenf = df[df['experimentType'] == 'TenF']
    if plot_type == 'scatter':
        ten_plot = ax.scatter(ten[feature], ten[outcome], c='b', label='Ten')
        tenf_plot = ax.scatter(tenf[feature], tenf[outcome], c='r',
                               label='TenF')
    elif plot_type == 'line':
        ten_plot = ax.scatter(ten[feature], ten[outcome], c='b', label='Ten')
        tenf_plot = ax.scatter(tenf[feature], tenf[outcome], c='r',
                               label='TenF')
        ax.plot(ten[feature], ten[outcome])
        ax.plot(tenf[feature], tenf[outcome])
    ax.set_title(outcome + ' vs ' + feature)
    ax.legend()
    output_dir = "./tenf_" + feature
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '/' + outcome.replace('/', '') + ".png"
    plt.savefig(output_file)


def plot_pred(X, y, model, outcome):
    indices = np.arange(X.shape[0])
    plt.figure(figsize=(20, 15))
    plt.plot(indices, model.predict(X), 'b*-', label='prediction')
    plt.plot(indices, y, 'r.', label='ground truth')
    plt.xlabel('Sample ID')
    plt.ylabel(outcome)
    plt.title(outcome)
    plt.legend()


def plot_ten_tenf(feature, outcome, ten, tenf):
    ten_tenf = ten.append(tenf, ignore_index=True).sort_values(by='sampleID')
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ind = np.arange(len(ten))
    width = 0.35
    rects1 = ax.bar(ind, ten[outcome], width, color='b')
    labels = ten[feature].append(tenf[feature])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels.values, rotation=75, fontsize=8)
    rects2 = ax.bar(ind + len(ten), tenf[outcome], width, color='r')
    ax.legend((rects1[0], rects2[0]), ('Original', 'Fused'))
    ax.set_ylabel(outcome)
    ax.set_xlabel(feature)
    output_file = "./ten_vs_tenf/" + outcome.replace('/', '') + ".png"
    plt.savefig(output_file)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(15, 10))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of training examples")
    plt.ylabel("r2 Score")

    indices = np.arange(y.shape[0])
    rng = np.random.RandomState(0)
    rng.shuffle(indices)

    # np.random.shuffle(indices)
    #
    X = X[indices]
    y = y[indices]

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='r2',
        n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # print 'test_scores: ', test_scores
    # print 'std: ', test_scores_std

    plt.grid()

    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")


def plot_all_learning_curve(estimator, data_matrix,
                            features=None, outcomes=None,
                            sample_rng=None, cv=3):
    features = features if features else data_matrix.features
    outcomes = outcomes if outcomes else data_matrix.outcomes
    sample_rng = sample_rng if sample_rng else data_matrix.sample_rng

    for outcome in outcomes:
        X_train, y_train = data_matrix.gen_train(features,
                                                 outcome,
                                                 sample_rng)
        # indices = np.arange(y_train.shape[0])
        # np.random.shuffle(indices)
        # plot_learning_curve(estimator, outcome, X_train[indices], y_train[indices], cv=cv)
        plot_learning_curve(estimator, outcome, X_train, y_train, cv=cv)


def plot_features_scores(outcome, features_scores, message1=None, message2=None,
                         shadow=None):
    labels = [x[0] + '\n' + '{:3.2f}'.format(x[3]) for x in features_scores]
    # labels = [x[0] + '\n' + x[3] for x in features_scores]
    scores = np.array([x[1] for x in features_scores])
    stds = np.array([x[2] for x in features_scores])
    plt.rc('font', size=16)
    plt.figure(figsize=(20, 10))
    plt.title(outcome)
    plt.xlabel('Features')
    plt.ylabel('Score')
    x = np.arange(len(scores))
    plt.xlim(-0.1, len(scores) - 0.9)

    if message1 is not None:
        xcoord = 1
        ycoord = scores[1]
        xcoord_text = xcoord + 0.05
        ycoord_text = scores[0] + (scores[1] - scores[0]) * 0.6
        plt.annotate(message1, xy=(xcoord, ycoord),
                     xytext=(xcoord_text, ycoord_text))
    if message2 is not None:
        xcoord = 2
        ycoord = scores[2]
        xcoord_text = xcoord + 0.05
        plt.annotate(message2, xy=(xcoord, ycoord),
                     xytext=(xcoord_text, ycoord_text))
    # plt.ylim(np.min(scores) - 0.5, np.max(scores) + 0.5)

    plt.ylim(-1, 1)
    plt.xticks(x, labels, rotation=30)
    plt.errorbar(x, scores, fmt='-o', yerr=stds, capsize=20,
                 markersize=20, color='yellowgreen')
    if shadow is not None:
        plt.fill_between(x, scores - stds, scores + stds, alpha=0.1,
                         color='yellowgreen')

    plt.tight_layout()
    plt.rc('font', size=20)
    plt.savefig('./figures/' + outcome + '_fs.png', dpi=300)
    plt.savefig('./figures/' + outcome + '_fs.pdf')


def plot_validation_curve(estimator_name, outcome_name, param_name, param_range,
                          train_scores, test_scores, shadow=None):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(15, 10))
    plt.title(
        'Validation Curve with ' + estimator_name + ' for ' + outcome_name)
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.rc('font', size=16)
    plt.grid()
    # If the x axis is linearly spaced
    if np.array_equal(np.linspace(param_range[0], param_range[-1],
                                  len(param_range)), param_range):
        plt.plot(param_range, train_scores_mean, '.-', label='Training Score',
                 color='darkorange')
        plt.plot(param_range, test_scores_mean, 'o-',
                 label='Cross-validation Score', color='navy')
    else:
        plt.semilogx(param_range, train_scores_mean, '.-',
                     label='Training Score', color='darkorange')
        plt.semilogx(param_range, test_scores_mean, 'o-',
                     label='Cross-validation Score', color='navy')

    if shadow is not None:
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color='darkorange')
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color='navy')
    _ = plt.legend(loc='upper right')
    plt.tight_layout()


def prediction_curve(mat, feature, outcome):
    ridge2 = make_pipeline(PolynomialFeatures(2), Ridge())
    X = mat[feature].as_matrix()[:, np.newaxis]
    y = mat[outcome].as_matrix()
    ridge2.fit(X, y)
    X_test = np.linspace(X.min(), X.max(), 50)[:, np.newaxis]
    y_test = ridge2.predict(X_test)
    plt.figure(figsize=(15, 10))
    plt.scatter(X, y, label='ground truth')
    plt.plot(X_test, y_test, color='r', label='prediction')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.xlabel(feature)
    plt.ylabel(outcome)
    r2 = r2_score(y, ridge2.predict(X))
    plt.title('R2: ' + str(r2))
