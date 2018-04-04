import random

from constants import *
import math
from itertools import *
from utils import *
from scipy.optimize import root
from scipy.special import digamma, polygamma, gammaln
import numpy as np


class NegBinomialOptimizationFailure(Exception):
    pass


class PoissonOptimizationFailure(Exception):
    pass


class ZeroDeltaException(NegBinomialOptimizationFailure):
    pass


OPTIMIZATION_LOGGING_LEVEL = 1


def dot_product(a, b):
    return sum(aa * bb for aa, bb in izip(a, b))


def neg_bionomial_log_likelihood_derivative(par, blocks, peak_posteriors, timepoint_idx,
                                            optimize_foreground=True):

    delta = par[0]
    betas = par[1:]

    d_delta = 0.
    n_betas = len(betas)
    d_betas = [0.] * n_betas

    for block_id in blocks:
        block_signal = blocks[block_id][FOREGROUND_SIGNAL][timepoint_idx]
        block_covariates = blocks[block_id][BLOCK_COVARIATES][timepoint_idx]
        block_peak_posteriors = peak_posteriors[block_id][timepoint_idx]

        for pos_idx in xrange(len(block_signal)):
            p_signal = block_signal[pos_idx]
            p_covariates = block_covariates[pos_idx]
            if optimize_foreground:
                weight = block_peak_posteriors[pos_idx]
            else:
                weight = 1 - block_peak_posteriors[pos_idx]

            mu = math.exp(dot_product(p_covariates, betas))
            xi = mu + delta

            d_delta += weight * (digamma(p_signal + delta) -
                                 digamma(delta) + math.log(delta) + 1 -
                                 math.log(xi) -
                                 delta / xi -
                                 p_signal / xi)

            for beta_idx in xrange(n_betas):
                d_betas[beta_idx] += weight * (p_signal * p_covariates[beta_idx] -
                                               (p_signal + delta) * p_covariates[beta_idx] * mu / xi)

    return np.array([d_delta] + d_betas)


def neg_bionomial_log_likelihood_derivative_of_betas_shared_beta(par,
                                                                 delta,
                                                                 peak_signal_array,
                                                                 peak_covariates_matrix,
                                                                 all_shared_peak_covariates_matrix,
                                                                 peak_posteriors_array,
                                                                 background_posteriors_array,
                                                                 all_signal_array,
                                                                 all_weights):

    shared_beta = par[0]
    peak_beta_0 = par[1]
    bgr_beta_0 = par[2]

    peak_betas = np.array([shared_beta, peak_beta_0])
    bgr_betas = np.array([shared_beta, bgr_beta_0])

    peak_mu = np.exp(np.dot(peak_betas, peak_covariates_matrix))
    bgr_mu = np.exp(np.dot(bgr_betas, peak_covariates_matrix))

    all_mu = np.concatenate((peak_mu, bgr_mu))

    d_shared_beta = np.sum(all_weights * (all_signal_array * all_shared_peak_covariates_matrix[0] -
                                         (all_signal_array + delta) * all_shared_peak_covariates_matrix[0] * all_mu /
                                          (all_mu + delta)))


    d_peak_beta_0 = np.sum(peak_posteriors_array * (peak_signal_array * peak_covariates_matrix[1] -
                                        (peak_signal_array + delta) * peak_covariates_matrix[1] * peak_mu / (peak_mu + delta)))

    d_bgr_beta_0 = np.sum(background_posteriors_array * (peak_signal_array * peak_covariates_matrix[1] -
                                        (peak_signal_array + delta) * peak_covariates_matrix[1] * bgr_mu / (bgr_mu + delta)))

    return np.array([d_shared_beta, d_peak_beta_0, d_bgr_beta_0])


def neg_bionomial_log_likelihood_derivative_of_betas(par,
                                                     delta,
                                                     peak_signal_array,
                                                     peak_covariates_matrix,
                                                     peak_posteriors):

    betas = par

    mu = np.exp(np.dot(betas, peak_covariates_matrix))
    xi = mu + delta
    d_betas = np.sum(peak_posteriors * (peak_signal_array * peak_covariates_matrix -
                                        (peak_signal_array + delta) * peak_covariates_matrix * mu / xi), axis=1)

    return d_betas


def neg_bionomial_log_likelihood_derivative_Jacobian(par, blocks, peak_posteriors, timepoint_idx,
                                                     optimize_foreground=True):

    delta = par[0]
    betas = par[1:]

    n_betas = len(betas)

    d_delta = [0.] * (n_betas + 1)
    d_betas = matrix(n_betas, n_betas + 1, default=0.)

    for block_id in blocks:
        block_signal = blocks[block_id][FOREGROUND_SIGNAL][timepoint_idx]
        block_covariates = blocks[block_id][BLOCK_COVARIATES][timepoint_idx]
        block_peak_posteriors = peak_posteriors[block_id][timepoint_idx]

        for pos_idx in xrange(len(block_signal)):
            p_signal = block_signal[pos_idx]
            p_covariates = block_covariates[pos_idx]

            if optimize_foreground:
                weight = block_peak_posteriors[pos_idx]
            else:
                weight = 1 - block_peak_posteriors[pos_idx]

            mu = math.exp(dot_product(p_covariates, betas))
            xi = mu + delta

            # update dF_{Delta_t}/dDelta_t
            d_delta[0] += weight * (polygamma(1, p_signal + delta) -
                                    polygamma(1, delta) +
                                    1. / delta -
                                    1 / xi -
                                    (mu - p_signal) / (xi ** 2))
            for beta_j in xrange(n_betas):

                # update dF_{Delta_t}/dBeta_{t,j}
                d_delta[1 + beta_j] += weight * p_covariates[beta_j] * mu * (
                    (delta + p_signal) / (xi ** 2) - 1. / xi)

                # update dF_{beta_{t,j}}/dDelta_t
                d_betas[beta_j][0] += weight * p_covariates[beta_j] * mu * (p_signal - mu) / (xi ** 2)

                # update dF_{Beta_{t,j}}/dBeta_{t,k}
                for beta_k in xrange(n_betas):
                    d_betas[beta_j][1 + beta_k] += weight * p_covariates[beta_j] * (p_signal + delta) * (
                        -p_covariates[beta_k] * mu * delta / (xi ** 2)
                    )

    return np.array([d_delta] + d_betas)


def optimize_Poisson_for_pair_of_NegBinomials(peak_signal_array, peak_covariates_matrix, peak_posteriors_array, background_posteriors_array, n_covariates):

    def score(par, peak_signal_array, peak_covariates_matrix, peak_posteriors_array):

        betas = par
        mu = np.exp(np.dot(betas, peak_covariates_matrix))

        return np.sum(peak_posteriors_array * peak_covariates_matrix * (peak_signal_array - mu), axis=1)

    predicted_means = []

    for foreground in [True, False]:

        if foreground:
            weights = peak_posteriors_array
        else:
            weights = background_posteriors_array

        solution = root(score,
                        np.array([1] * n_covariates),
                        args=(peak_signal_array, peak_covariates_matrix, weights))

        if not solution.success:
            echo("Error estimating the poisson parameters in Signal NBs:", solution.message, level=OPTIMIZATION_LOGGING_LEVEL)
            raise PoissonOptimizationFailure

        betas = solution.x

        mu = np.exp(np.dot(betas, peak_covariates_matrix))

        predicted_means.append(mu)

    weights = np.concatenate((peak_posteriors_array, background_posteriors_array))

    y = np.concatenate((peak_signal_array, peak_signal_array))

    mu = np.concatenate(tuple(predicted_means))

    delta = delta_max_likelihood(y, mu, weights)

    return delta, predicted_means[0], predicted_means[1]


def optimize_Poisson_for_pair_of_NegBinomials_shared_beta(peak_signal_array,
                                                          all_signal_array,
                                                          peak_covariates_matrix,
                                                          all_shared_peak_covariates_matrix,
                                                          peak_posteriors_array,
                                                          background_posteriors_array,
                                                          all_posteriors_array,
                                                          n_covariates):

    def score(par,
              peak_signal_array,
              all_signal_array,
              peak_covariates_matrix,
              all_shared_peak_covariates_matrix,
              peak_posteriors_array,
              background_posteriors_array,
              all_posteriors_array):

        shared_beta = par[0]
        peak_beta_0 = par[1]
        bgr_beta_0 = par[2]

        peak_betas = [shared_beta, peak_beta_0]
        bgr_betas = [shared_beta, bgr_beta_0]

        peak_mu = np.exp(np.dot(peak_betas, peak_covariates_matrix))
        bgr_mu = np.exp(np.dot(bgr_betas, peak_covariates_matrix))

        all_mu = np.concatenate((peak_mu, bgr_mu))

        d_shared_beta = np.sum(all_posteriors_array * all_shared_peak_covariates_matrix[0] * (all_signal_array - all_mu))
        d_peak_beta_0 = np.sum(peak_posteriors_array * peak_covariates_matrix[1] * (peak_signal_array - peak_mu))
        d_bgr_beta_0 = np.sum(background_posteriors_array * peak_covariates_matrix[1] * (peak_signal_array - bgr_mu))
        return [d_shared_beta, d_peak_beta_0, d_bgr_beta_0]

    init_betas = np.array([1] * (2 * n_covariates - 1))

    solution = root(score,
                    init_betas,
                    args=(peak_signal_array,
                          all_signal_array,
                          peak_covariates_matrix,
                          all_shared_peak_covariates_matrix,
                          peak_posteriors_array,
                          background_posteriors_array,
                          all_posteriors_array))

    if not solution.success:
        echo("Error estimating the poisson parameters in Signal NBs:", solution.message, level=OPTIMIZATION_LOGGING_LEVEL)
        raise PoissonOptimizationFailure

    shared_beta, peak_beta_0, bgr_beta_0 = solution.x

    peak_mu = np.exp(np.dot([shared_beta, peak_beta_0], peak_covariates_matrix))
    bgr_mu = np.exp(np.dot([shared_beta, bgr_beta_0], peak_covariates_matrix))

    all_mu = np.concatenate((peak_mu, bgr_mu))

    delta = delta_max_likelihood(all_signal_array, all_mu, all_posteriors_array)

    return delta, peak_mu, bgr_mu


def optimize_a_pair_of_NegBinomials_jointly(peak_signal_array, peak_covariates_matrix, peak_posteriors_array,
                                            timepoint_idx, init_fgr_betas, init_bgr_betas, n_covariates):

    cur_likelihood = -float('inf')

    MAX_ITERATIONS = 25
    MIN_DIFF = 0.0001220703125

    diff = 1

    background_posteriors_array = 1 - peak_posteriors_array
    cur_fgr_betas = np.array(init_fgr_betas)
    cur_bgr_betas = np.array(init_bgr_betas)


    try:
        init_delta, predicted_peak_means, predicted_background_means = optimize_Poisson_for_pair_of_NegBinomials(peak_signal_array,
                                                                                                                 peak_covariates_matrix,
                                                                                                                 peak_posteriors_array,
                                                                                                                 background_posteriors_array,
                                                                                                                 n_covariates)
    except ZeroDeltaException:
        echo('WARNING: Zero delta exception has occurred while estimating initial Poisson for SIGNAL NBs for time point', timepoint_idx, level=OPTIMIZATION_LOGGING_LEVEL)
        return None, None, None, None, timepoint_idx
    except PoissonOptimizationFailure:
        echo('WARNING: Poisson optimization failure has occurred while estimating SIGNAL NBs for time point', timepoint_idx, level=OPTIMIZATION_LOGGING_LEVEL)
        return None, None, None, None, timepoint_idx

    cur_fgr_delta = cur_bgr_delta = init_delta

    all_posteriors_array = np.concatenate((peak_posteriors_array, background_posteriors_array))
    all_signal_array = np.concatenate((peak_signal_array, peak_signal_array))

    for iteration_idx in xrange(MAX_ITERATIONS):

        if diff < MIN_DIFF:
            break

        solution = root(neg_bionomial_log_likelihood_derivative_of_betas,
                        cur_fgr_betas,
                        args=(cur_fgr_delta, peak_signal_array, peak_covariates_matrix, peak_posteriors_array))

        if not solution.success:
            echo("Error estimating the negative binomial parameters in foreground signal NBs:", solution.message, level=OPTIMIZATION_LOGGING_LEVEL)
            return None, None, None, None, timepoint_idx

        cur_fgr_betas = solution.x

        solution = root(neg_bionomial_log_likelihood_derivative_of_betas,
                        cur_bgr_betas,
                        args=(cur_bgr_delta, peak_signal_array, peak_covariates_matrix, background_posteriors_array))

        if not solution.success:
            echo("Error estimating the negative binomial parameters in background signal NBs:", solution.message, level=OPTIMIZATION_LOGGING_LEVEL)
            return None, None, None, None, timepoint_idx

        cur_bgr_betas = solution.x

        try:
            cur_fgr_delta = cur_bgr_delta = delta_max_likelihood(all_signal_array,
                                                                 np.concatenate((predicted_peak_means,
                                                                                 predicted_background_means)),
                                                                 all_posteriors_array)
        except ZeroDeltaException:
            echo('WARNING: Zero delta exception has occurred while estimating SIGNAL NBs for time point', timepoint_idx, level=OPTIMIZATION_LOGGING_LEVEL)
            return None, None, None, None, timepoint_idx

        predicted_peak_means = np.exp(np.dot(cur_fgr_betas, peak_covariates_matrix))
        predicted_background_means = np.exp(np.dot(cur_bgr_betas, peak_covariates_matrix))

        def log_likelihood(w, delta, y, mu):
            return np.sum(w * (gammaln(delta + y)
                               - gammaln(delta)
                               + delta * np.log(delta)
                               - (delta + y) * np.log(mu + delta)
                               + y * np.log(mu)))

        prev_likelihood = cur_likelihood

        cur_likelihood = log_likelihood(peak_posteriors_array,
                                        cur_fgr_delta,
                                        peak_signal_array,
                                        predicted_peak_means) + \
                         log_likelihood(background_posteriors_array,
                                        cur_bgr_delta,
                                        peak_signal_array,
                                        predicted_background_means)

        diff = cur_likelihood - prev_likelihood

    return cur_fgr_delta, cur_fgr_betas, cur_bgr_delta, cur_bgr_betas, timepoint_idx


def optimize_a_pair_of_NegBinomials_jointly_shared_beta(peak_signal_array,
                                                        peak_covariates_matrix,
                                                        peak_posteriors_array,
                                                        timepoint_idx,
                                                        init_fgr_betas,
                                                        init_bgr_betas,
                                                        n_covariates):

    cur_likelihood = -float('inf')

    MAX_ITERATIONS = 25
    MIN_DIFF = 0.0001220703125

    diff = 1

    background_posteriors_array = 1 - peak_posteriors_array

    cur_betas = np.array([init_fgr_betas[0], init_fgr_betas[1], init_bgr_betas[1]])

    cur_fgr_betas = np.array([cur_betas[0], cur_betas[1]])
    cur_bgr_betas = np.array([cur_betas[0], cur_betas[2]])

    all_shared_peak_covariates_matrix = np.concatenate((peak_covariates_matrix, peak_covariates_matrix), axis=1)
    all_posteriors_array = np.concatenate((peak_posteriors_array, background_posteriors_array))
    all_signal_array = np.concatenate((peak_signal_array, peak_signal_array))

    try:
        init_delta, predicted_peak_means, predicted_background_means = \
            optimize_Poisson_for_pair_of_NegBinomials_shared_beta(peak_signal_array,
                                                                  all_signal_array,
                                                                  peak_covariates_matrix,
                                                                  all_shared_peak_covariates_matrix,
                                                                  peak_posteriors_array,
                                                                  background_posteriors_array,
                                                                  all_posteriors_array,
                                                                  n_covariates)
    except ZeroDeltaException:
        echo('WARNING: Zero delta exception has occurred while estimating initial Poisson for SIGNAL NBs for time point', timepoint_idx, level=OPTIMIZATION_LOGGING_LEVEL)
        return None, None, None, None, timepoint_idx
    except PoissonOptimizationFailure:
        echo('WARNING: Poisson optimization failure has occurred while estimating SIGNAL NBs for time point', timepoint_idx, level=OPTIMIZATION_LOGGING_LEVEL)
        return None, None, None, None, timepoint_idx

    cur_fgr_delta = cur_bgr_delta = init_delta

    for iteration_idx in xrange(MAX_ITERATIONS):

        if diff < MIN_DIFF:
            break

        solution = root(neg_bionomial_log_likelihood_derivative_of_betas_shared_beta,
                        cur_betas,
                        args=(cur_fgr_delta,
                              peak_signal_array,
                              peak_covariates_matrix,
                              all_shared_peak_covariates_matrix,
                              peak_posteriors_array,
                              background_posteriors_array,
                              all_signal_array,
                              all_posteriors_array))

        if not solution.success:
            echo("Error estimating the negative binomial parameters in foreground signal NBs:", solution.message, level=OPTIMIZATION_LOGGING_LEVEL)
            return None, None, None, None, timepoint_idx

        cur_betas = solution.x
        cur_fgr_betas = np.array([cur_betas[0], cur_betas[1]])
        cur_bgr_betas = np.array([cur_betas[0], cur_betas[2]])

        try:
            cur_fgr_delta = cur_bgr_delta = delta_max_likelihood(all_signal_array,
                                                                 np.concatenate((predicted_peak_means,
                                                                                 predicted_background_means)),
                                                                 all_posteriors_array)
        except ZeroDeltaException:
            echo('WARNING: Zero delta exception has occurred while estimating SIGNAL NBs for time point', timepoint_idx, level=OPTIMIZATION_LOGGING_LEVEL)
            return None, None, None, None, timepoint_idx

        predicted_peak_means = np.exp(np.dot(cur_fgr_betas, peak_covariates_matrix))
        predicted_background_means = np.exp(np.dot(cur_bgr_betas, peak_covariates_matrix))

        def log_likelihood(w, delta, y, mu):
            return np.sum(w * (gammaln(delta + y)
                               - gammaln(delta)
                               + delta * np.log(delta)
                               - (delta + y) * np.log(mu + delta)
                               + y * np.log(mu)))

        prev_likelihood = cur_likelihood

        cur_likelihood = log_likelihood(peak_posteriors_array,
                                        cur_fgr_delta,
                                        peak_signal_array,
                                        predicted_peak_means) + \
                         log_likelihood(background_posteriors_array,
                                        cur_bgr_delta,
                                        peak_signal_array,
                                        predicted_background_means)

        diff = cur_likelihood - prev_likelihood

    if cur_fgr_delta < 0.001:
        raise NegBinomialOptimizationFailure

    return cur_fgr_delta, cur_fgr_betas, cur_bgr_delta, cur_bgr_betas, timepoint_idx


def optimize_Poisson(y, x, weights, n_covariates):

    def score(par, y, x, weights):

        betas = par
        mu = np.exp(np.dot(betas, x))

        return np.sum(weights * x * (y - mu), axis=1)

    solution = root(score,
                    np.array([1] * n_covariates),
                    args=(y, x, weights))

    if not solution.success:
        echo("Error estimating the poisson parameters in dynamics NBs:", solution.message, level=OPTIMIZATION_LOGGING_LEVEL)
        raise PoissonOptimizationFailure

    betas = solution.x

    predicted_means = np.exp(np.dot(betas, x))

    delta = delta_max_likelihood(y, predicted_means, weights)

    return delta, predicted_means


def optimize_NegBinomial(y, x, weights, init_betas):

    y = np.array(y)
    x = np.array(x)
    weights = np.array(weights)

    cur_betas = np.array(init_betas)

    cur_likelihood = -float('inf')

    MAX_ITERATIONS = 25
    MIN_DIFF = 0.0001220703125

    diff = 1

    cur_delta, predicted_diff_means = optimize_Poisson(y, x, weights, 1)

    for iteration_idx in xrange(MAX_ITERATIONS):

        if diff < MIN_DIFF:
            break

        solution = root(neg_bionomial_log_likelihood_derivative_of_betas,
                        cur_betas,
                        args=(cur_delta, y, x, weights))

        if not solution.success:
            raise NegBinomialOptimizationFailure

        cur_betas = solution.x

        cur_delta = delta_max_likelihood(y, predicted_diff_means, weights)

        predicted_diff_means = np.exp(np.dot(cur_betas, x))

        def log_likelihood(w, delta, y, mu):
            return np.sum(w * (gammaln(delta + y)
                               - gammaln(delta)
                               + delta * np.log(delta)
                               - (delta + y) * np.log(mu + delta)
                               + y * np.log(mu)))

        prev_likelihood = cur_likelihood

        cur_likelihood = log_likelihood(weights, cur_delta, y, predicted_diff_means)

        diff = cur_likelihood - prev_likelihood

    if cur_delta < 0.001:
        raise NegBinomialOptimizationFailure

    return cur_delta, cur_betas


def delta_max_likelihood(y, mu, weights):
    # This code was adapted from the MASS R package

    n = sum(weights)

    def score(th, mu, y, w):
        return np.sum(w * (digamma(th + y) - digamma(th) + np.log(th) +
                           1 - np.log(th + mu) - (y + th) / (mu + th)))

    def info(th, mu, y, w):
        return np.sum(w * (- polygamma(1, th + y) + polygamma(1, th) - 1 / th +
                           2 / (mu + th) - (y + th) / (mu + th) ** 2))

    t0 = n / np.sum(weights * (y / mu - 1) ** 2)

    iteration_idx = 0
    diff = 1

    MAX_ITERATIONS = 25
    MAX_DIFF = 0.0001220703125
    if t0 < MAX_DIFF:
        t0 = 10 * MAX_DIFF

    while iteration_idx < MAX_ITERATIONS - 1 and abs(diff) > MAX_DIFF:
        iteration_idx += 1
        t0 = abs(t0)
        diff = score(t0, mu, y, weights) / info(t0, mu, y, weights)

        t0 = t0 + diff

    if t0 <= 0 or math.isnan(t0):

        echo('Warning: Negative binomial delta estimated to be <= 0 or nan:' + str(t0), level=OPTIMIZATION_LOGGING_LEVEL)
        import traceback
        if hasattr(open_log, 'logfile'):
            traceback.print_stack(file=open_log.logfile)
        raise ZeroDeltaException

    MAX_DELTA = 100000
    if t0 > MAX_DELTA:
        t0 = MAX_DELTA
    return t0


def optimize_nb_R(y, x, weights):
    from rpy2 import robjects as ro
    import rpy2.rlike.container as rlc
    from rpy2.robjects.packages import importr
    ro.r['options'](warn=1)
    mass = importr("MASS")

    x_float_vector = [ro.FloatVector(xx) for xx in x]

    y_float_vector = ro.FloatVector(y)

    weights_float_vector = ro.FloatVector(weights)

    names = ['v' + str(i) for i in xrange(len(x_float_vector))]
    d = rlc.TaggedList(x_float_vector + [y_float_vector], names + ['y'])
    data = ro.DataFrame(d)
    formula = 'y ~ ' + '+ '.join(names) + ' - 1'

    try:
        fit_res = mass.glm_nb(formula=ro.r(formula), data=data, weights=weights_float_vector)
    except:
        return NegBinomialOptimizationFailure

    return fit_res.rx2('theta')[0], list(fit_res.rx2('coefficients'))
