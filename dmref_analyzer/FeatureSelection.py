from sklearn.feature_selection import RFE
from dmref_analyzer import ModelSelection
from dmref_analyzer import PlotGenerator

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import numpy as np


def select_all(estimator, data_matrix, n_features):
  features = data_matrix.features
  selector = RFE(estimator, n_features, step=1)
  for outcome in data_matrix.outcomes:
    print outcome
    X_train, y_train = data_matrix.gen_train(data_matrix.features,
                                             outcome,
                                             data_matrix.sample_rng)
    selector = selector.fit(X_train, y_train)
    print [features[i] for i in np.arange(len(features)) if
           selector.support_[i]]


def wrapper_forward_selection(estimator_name,
                              estimator,
                              param_name, param_range,
                              repetition, cv,
                              data_matrix, features, outcome,
                              sample_rng=None,
                              plot=True):
  features_not_included = features[:]
  features_included = []
  features_scores = []

  if sample_rng is None:
    sample_rng = data_matrix.sample_rng

  while len(features_not_included) > 0:
    best_score = -float('inf')
    best_feature = None
    best_param = None
    for feature in features_not_included:
      candidate_features = features_included + [feature]

      if param_name is not None:
        train_scores, test_scores = \
          ModelSelection.tune_param(None, estimator,
                                    param_name, param_range,
                                    repetition, cv,
                                    data_matrix,
                                    candidate_features, outcome,
                                    sample_rng,
                                    False)
        index = np.mean(test_scores, axis=1).argmax()
        score = np.mean(test_scores, axis=1)[index]
        score_std = np.std(test_scores, axis=1)[index]
        param = param_range[index]
        if score > best_score:
          best_score = score
          best_score_std = score_std
          best_feature = feature
          best_param = param
          best_scores = test_scores[index]
      else:
        test_scores_concat = None
        X_train_orig, y_train_orig = \
          data_matrix.gen_train(features, outcome)
        indices = np.arange(y_train_orig.shape[0])
        for rep in np.arange(repetition):
          np.random.shuffle(indices)
          X_train = X_train_orig[indices]
          y_train = y_train_orig[indices]
          test_scores = cross_val_score(estimator, X_train, y_train,
                                        cv=cv)
          if test_scores_concat is None:
            test_scores_concat = test_scores
          else:
            test_scores_concat = np.concatenate(
              (test_scores_concat, test_scores))
        test_scores = test_scores_concat
        score = np.mean(test_scores)
        score_std = np.std(test_scores)
        if score > best_score:
          best_score = score
          best_score_std = score_std
          best_feature = feature
          best_param = -1

    features_included.append(best_feature)
    features_not_included.remove(best_feature)
    # using [:] to make a copy of features_included
    features_scores.append(
      (best_feature, best_score, best_score_std, best_param, best_scores))
  if plot:
    # if estimator_name == 'Ridge2':
    #     terms = ['x0', 'x1', 'x2', 'x1^2', 'x1x2', 'x2^2']
    #     best_features = [features_scores[0][0], features_scores[1][0]]
    #     alpha = features_scores[1][3]
    #     X, y = data_matrix.gen_train(best_features, outcome)
    #     best_ridge2 = make_pipeline(PolynomialFeatures(2),
    #                                 Ridge(alpha=alpha))
    #     best_ridge2.fit(X, y)
    #     coefs = best_ridge2.named_steps['ridge'].coef_.flatten()
    #     pairs = zip(coefs, terms)
    #     s2 = ''
    #     for coef, term in pairs:
    #         s2 += '{:3.2f}'.format(coef) + ', ' + term + '\n'
    #     # terms = ['x0', 'x1', 'x2', 'x3', 'x1^2', 'x1x2', 'x1x3', 'x2^2', 'x2x3', 'x3^2']
    #     # best_features = [features_scores[0][0], features_scores[1][0], features_scores[2][0]]
    #     # alpha = features_scores[2][3]
    #     # X, y = data_matrix.gen_train(best_features, outcome)
    #     # best_ridge2 = make_pipeline(PolynomialFeatures(2), Ridge(alpha=alpha))
    #     # best_ridge2.fit(X, y)
    #     # coefs = best_ridge2.named_steps['ridge'].coef_.flatten()
    #     # pairs = zip(coefs, terms)
    #     # s3 = ''
    #     # for coef, term in pairs:
    #     #     s3 += '{:3.2f}'.format(coef) + ', ' + term + '\n'
    #     PlotGenerator.plot_features_scores(outcome, features_scores, s2, None)
    # else:
    title = estimator_name + '_' + outcome
    PlotGenerator.plot_features_scores(title, features_scores)

  return features_scores
