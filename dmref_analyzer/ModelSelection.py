from sklearn.model_selection import validation_curve
import numpy as np
import dmref_analyzer.PlotGenerator as pg


def tune_param(estimator_name, estimator, param_name, param_range,
               repetition, cv, data_matrix, features, outcome, sample_rng,
               plot=True):


  train_scores_concat = None
  test_scores_concat = None

  X_train_orig, y_train_orig = \
    data_matrix.gen_train(features, outcome, sample_rng)

  indices = np.arange(y_train_orig.shape[0])

  for rep in np.arange(repetition):
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    X_train = X_train_orig[indices]
    y_train = y_train_orig[indices]

    if len(features) > 1:
      train_scores, test_scores = validation_curve(
        estimator, X_train, y_train, param_name=param_name,
        param_range=param_range,
        cv=cv, scoring='r2', n_jobs=-1)
    elif len(features) == 1:
      train_scores, test_scores = validation_curve(
        estimator, X_train.reshape(-1, 1), y_train,
        param_name=param_name, param_range=param_range,
        cv=cv, scoring='r2', n_jobs=-1)
    if train_scores_concat is None:
      train_scores_concat = train_scores
      test_scores_concat = test_scores
    else:
      train_scores_concat = np.concatenate(
        (train_scores_concat, train_scores), axis=1)
      test_scores_concat = np.concatenate(
        (test_scores_concat, test_scores), axis=1)

  if plot:
    pg.plot_validation_curve(estimator_name,
                             outcome,
                             param_name,
                             param_range,
                             train_scores_concat, test_scores_concat)

  return train_scores_concat, test_scores_concat
