"""Module containing methods to fit various forests."""
import sys

import numpy as np
import pandas as pd  # noqa: F401
import rpy2  # noqa: F401


def causal_forest(
    x,
    y,
    w,
    y_hat=None,
    w_hat=None,
    num_trees=2000,
    sample_weights=None,
    clusters=None,
    equalize_cluster_weights=False,
    sample_fraction=0.5,
    mtry=None,
    min_node_size=5,
    honesty=True,
    honesty_fraction=0.5,
    honesty_prune_leaves=True,
    alpha=0.05,
    imbalance_penalty=0,
    stabilize_splits=True,
    ci_group_size=2,
    tune_parameters="none",
    tune_num_trees=200,
    tune_num_reps=50,
    tune_num_draws=1000,
    compute_oob_predictions=True,
    orthog_boosting=False,
    num_threads=None,
    seed=None,
):
    r"""Fit a causal forest.

    Trains a causal forest that can be used to estimate conditional average treatment
    effects tau(X). When the treatment assignment W is binary and unconfounded, we have
    tau(X) = E[Y(1) - Y(0) | X = x], where Y(0) and Y(1) are potential outcomes
    corresponding to the two possible treatment states.  When W is continuous, we
    effectively estimate an average partial effect Cov[Y, W | X = x] / Var[W | X = x],
    and interpret it as a treatment effect given unconfoundedness.

    For a detailed description of honesty, honesty_fraction, honesty_prune_leaves, and
    recommendations for parameter tuning, see the grf
    https://grf-labs.github.io/grf/REFERENCE.html#honesty-honesty-fraction-honesty-prune-leaves

    Args:
        x (pd.DataFrame or np.ndarray): The covariates used in the causal regression.

        y (pd.Series of np.ndarray): The outcome (must be a numeric vector with no NAs).

        w (pd.Series or np.ndarray): The treatment assignment (must be a binary or real
            numeric vector with no NAs).

        y_hat (pd.Series or np.ndarray): Estimates of the expected responses E[Y | Xi],
            marginalizing over treatment. If Y_hat = None, these are estimated using a
            separate regression forest. See section 6.1.1 of the GRF paper for further
            discussion of this quantity. Default is None.

        w_hat (pd.Series or np.ndarray): Estimates of the treatment propensities E[W |
            Xi]. If W_hat = None, these are estimated using a separate regression
            forest. Default is None.

        num_trees (int): Number of trees grown in the forest. Note: Getting accurate
            confidence intervals generally requires more trees than getting accurate
            predictions. Default is 2000.

        sample_weights (list-like): Weights given to each sample in estimation.  If
            None, each observation receives the same weight.  Note: To avoid introducing
            confounding, weights should be independent of the potential outcomes given
            X.  Default is None.

        clusters (list-like): Vector of integers or factors specifying which cluster
            each observation corresponds to. Default is None (ignored).

        equalize_cluster_weights (list-like): If False, each unit is given the same
            weight (so that bigger clusters get more weight). If True, each cluster is
            given equal weight in the forest. In this case, during training, each tree
            uses the same number of observations from each drawn cluster: If the
            smallest cluster has K units, then when we sample a cluster during training,
            we only give a random K elements of the cluster to the tree-growing
            procedure. When estimating average treatment effects, each observation is
            given weight 1/cluster size, so that the total weight of each cluster is the
            same. Note that, if this argument is False, sample weights may also be
            directly adjusted via the sample.weights argument. If this argument is True,
            sample_weights must be set to None. Default is False.

        sample_fraction (float): Fraction of the data used to build each tree.  Note: If
            honesty = True, these subsamples will further be cut by a factor of
            honesty_fraction Default is 0.5.

        mtry (int): Number of variables tried for each split. Default is \eqn{\sqrt p +
            20} where p is the number of variables.

        min_node_size (int): A target for the minimum number of observations in each
            tree leaf. Note that nodes with size smaller than min_node_size can occur,
            as in the original randomForest package.  Default is 5.  honesty (bool):
            Whether to use honest splitting (i.e., sub-sample splitting). Default is
            True.

        honesty_fraction (float): The fraction of data that will be used for determining
            splits if honesty = True. Corresponds to set J1 in the notation of the
            paper.  Default is 0.5 (i.e. half of the data is used for determining
            splits).

        honesty_prune_leaves (bool): If True, prunes the estimation sample tree such
            that no leaves are empty. If False, keep the same tree as determined in the
            splits sample (if an empty leave is encountered, that tree is skipped and
            does not contribute to the estimate). Setting this to False may improve
            performance on small/marginally powered data, but requires more trees (note:
            tuning does not adjust the number of trees).  Only applies if honesty is
            enabled. Default is True.

        alpha (float): A tuning parameter that controls the maximum imbalance of a
            split. Default is 0.05.

        imbalance_penalty (float): A tuning parameter that controls how harshly
            imbalanced splits are penalized. Default is 0.

        stabilize_splits (bool): Whether or not the treatment should be taken into
            account when determining the imbalance of a split. Default is True.

        ci_group_size (int): The forest will grow ci_group_size trees on each subsample.
            In order to provide confidence intervals, ci_group_size must be at least 2.
            Default is 2.

        tune_parameters (list-like): A vector of parameter names to tune.  If "all": all
            tunable parameters are tuned by cross-validation. The following parameters
            are tunable: ("sample_fraction", "mtry", "min_node_size",
            "honesty_fraction", "honesty_prune_leaves", "alpha", "imbalance_penalty").
            If honesty is False the honesty.* parameters are not tuned.  Default is
            "none" (no parameters are tuned).

        tune_num_trees (int): The number of trees in each 'mini forest' used to fit the
            tuning model. Default is 200.

        tune_num_reps (int): The number of forests used to fit the tuning model. Default
            is 50.

        tune_num_draws (int): The number of random parameter values considered when
            using the model to select the optimal parameters. Default is 1000.

        compute_oob_predictions (bool): Whether OOB predictions on training set should
            be precomputed. Default is True.

        orthog_boosting (bool): [experimental] If True, then when Y_hat = None or W_hat
            is None, the missing quantities are estimated using boosted regression
            forests.  The number of boosting steps is selected automatically. Default is
            False.

        num_threads (int): Number of threads used in training. By default, the number of
            threads is set to the maximum hardware concurrency.

        seed (int): The seed of the C++ random number generator.

    Returns:
        A trained causal forest object. If tune.parameters is enabled, then tuning
        information will be included through the `tuning_output` attribute.

    Examples:
        # Train a causal forest.
        n <- 500
        p <- 10
        X <- matrix(rnorm(n * p), n, p)
        W <- rbinom(n, 1, 0.5)
        Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)
        c.forest <- causal_forest(X, Y, W)

        # Predict using the forest.
        X.test <- matrix(0, 101, p)
        X.test[, 1] <- seq(-2, 2, length.out = 101)
        c.pred <- predict(c.forest, X.test)

        # Predict on out-of-bag training samples.
        c.pred <- predict(c.forest)

        # Predict with confidence intervals; growing more trees is now recommended.
        c.forest <- causal_forest(X, Y, W, num.trees = 4000)
        c.pred <- predict(c.forest, X.test, estimate.variance = True)  # noqa: RST203

        # In some examples, pre-fitting models for Y and W separately may
        # be helpful (e.g., if different models use different covariates).
        # In some applications, one may even want to get Y.hat and W.hat
        # using a completely different method (e.g., boosting).
        n <- 2000
        p <- 20
        X <- matrix(rnorm(n * p), n, p)
        TAU <- 1 / (1 + exp(-X[, 3]))
        W <- rbinom(n, 1, 1 / (1 + exp(-X[, 1] - X[, 2])))
        Y <- pmax(X[, 2] + X[, 3], 0) + rowMeans(X[, 4:6]) / 2 + W * TAU + rnorm(n)

        forest.W <- regression_forest(X, W, tune.parameters = "all")
        W.hat <- predict(forest.W)$predictions

        forest.Y <- regression_forest(X, Y, tune.parameters = "all")
        Y.hat <- predict(forest.Y)$predictions

        forest.Y.varimp <- variable_importance(forest.Y)

        # Note: Forests may have a hard time when trained on very few variables
        # (e.g., ncol(X) = 1, 2, or 3). We recommend not being too aggressive
        # in selection.
        selected.vars <- which(forest.Y.varimp / mean(forest.Y.varimp) > 0.2)

        tau.forest <- causal_forest(X[, selected.vars], Y, W,
          W.hat = W.hat, Y.hat = Y.hat,
          tune.parameters = "all"
        )
        tau.hat <- predict(tau.forest)$predictions

    """
    # assert arguments

    # assign defaults
    num_rows_features, num_cols_features = x.shape
    mtry = (
        np.min(np.ceil(np.sqrt(num_cols_features + 20)), num_cols_features)
        if mtry is None
        else mtry
    )
    seed = int(np.random.uniform(0, sys.maxsize, 1)) if seed is None else seed

    # transform python arguments to R arguments

    # call r function
    pass
