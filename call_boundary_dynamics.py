import argparse
import gzip
import traceback
from itertools import izip
import os
import pprint
import random
import sys
import math
import itertools
import cPickle as pickle
from multiprocessing import Pool
from utils import *
from constants import *
from optimize import *
import numpy as np

import ctypes
clib = ctypes.CDLL(os.path.join(os.path.split(__file__)[0],
                                'C_call_boundary_dynamics.so'))


def log_poisson_pmf(k, Lambda):

    if Lambda == 0:
        return 0 if k == 0 else -float('inf')

    key = (k, Lambda)
    if key not in log_poisson_pmf.cache:
        log_poisson_pmf.cache[key] = k * math.log(Lambda) - math.lgamma(k + 1.0) - Lambda
    return log_poisson_pmf.cache[key]
log_poisson_pmf.cache = {}

__author__ = 'Fiziev'

NO_SIGNAL = 0
PEAK = 1

ACTIVE = 'active'
BACKGROUND = 'background'

EXPAND = 'E'
CONTRACT = 'C'
STEADY = 'S'

LEFT = 'left'
RIGHT = 'right'

LAMBDA = 'lambda'
SCALE = 'scale'

EMISSION_PARAMS = 'emission_params'
DYNAMICS_PARAMS = 'dynamics_params'

# log2 = lambda x: math.log(x, 2)


class DecreasingLikelihoodException(Exception):
    def __init__(self, message, delta_log_likelihood, blocks):
        # Call the base class constructor with the parameters it needs
        super(DecreasingLikelihoodException, self).__init__(message)

        self.delta_log_likelihood = delta_log_likelihood
        self.blocks = blocks


class UnderflowException(Exception):
    pass


def read_bed(bed_fname, bin_size, filter_min=0, filter_max=None):
    regions = {}
    echo('Reading:', bed_fname)
    with (gzip.open(bed_fname) if bed_fname.endswith('.gz') else open(bed_fname)) as in_f:
        for line in in_f:

            if line.startswith('#') or line.startswith('track'):
                continue

            buf = line.strip().split()
            chrom, start, end, reg_id = buf[:4]

            start = int(start)
            end = int(end)

            if filter_min < end - start:
                if filter_max is None or end - start <= filter_max:
                    regions[reg_id] = (chrom,
                                       (start + bin_size / 2) / bin_size,
                                       1 + (end - 1 - bin_size / 2) / bin_size)

    return regions


def read_block_wig_signal(input_fnames, block_boundaries, bin_size):

    block_signal = dict((reg_id, [[] for _ in xrange(len(input_fnames))]) for reg_id in block_boundaries)

    for t_idx, reps_fnames in enumerate(input_fnames):

        for signal_fname in reps_fnames:
            echo('Reading signal from:', signal_fname)

            with gzip.open(signal_fname) if signal_fname.endswith('.gz') else open(signal_fname) as in_f:

                wig_signal = {}
                chrom = None
                span = None

                for line in in_f:

                    if line.startswith('track'):
                        continue

                    if line.startswith('fixedStep'):
                        chrom = re.search(r'chrom=(\w+)', line).group(1)
                        wig_signal[chrom] = []
                        span = int(re.search(r'span=(\d+)', line).group(1))
                        continue

                    wig_signal[chrom].append(float(line))

            rescale_step = bin_size / span
            if rescale_step == 0:
                print 'ERROR: span and bin_size are incompatible:', span, bin_size
                exit(1)

            # rescale the signal if necessary
            for chrom in wig_signal:
                wig_signal[chrom] = [mean(wig_signal[chrom][bin_pos: bin_pos + rescale_step])
                                        for bin_pos in xrange(0, len(wig_signal[chrom]), rescale_step)]

            # extract the signal for all regions
            for reg_id in block_boundaries:
                chrom, start, end = block_boundaries[reg_id]
                block_signal[reg_id][t_idx].append(map(lambda s: math.log(s + 1, 2), wig_signal[chrom][start: end]))

    return block_signal


class ClusterModel:

    def __init__(self,
                 n_timepoints,
                 out_prefix,
                 n_threads,
                 max_region_length,
                 # min_lambda,
                 # max_lambda,
                 bin_size,
                 n_covariates):

        self.n_threads = n_threads
        self.bin_size = bin_size
        self.n_timepoints = n_timepoints

        self.max_region_length = max_region_length
        # self.min_lambda = min_lambda
        # self.max_lambda = max_lambda
        self.n_covariates = n_covariates

        self.dynamics = [STEADY, EXPAND, CONTRACT]
        self.dynamics_idx = dict((d, self.dynamics.index(d)) for d in self.dynamics)

        self.n_dynamics = len(self.dynamics)

        self.model_fname = out_prefix + '.model.pickle'

        self.reset_model()

        echo('Model is created', level=ECHO_TO_LOGFILE)
        # self.print_model()

    def reset_model(self):
        self.prev_total_likelihood = None

        n_timepoints = self.n_timepoints
        n_covariates = self.n_covariates

        self.fdr_threshold_for_decoding = [None] * self.n_timepoints
        self.foreground_betas = matrix(n_timepoints, n_covariates, default=1.)
        self.background_betas = matrix(n_timepoints, n_covariates, default=1.)

        self.foreground_delta = [1.] * n_timepoints
        self.background_delta = [1.] * n_timepoints

        self.dynamics_params = [[[0, 0] if dyn == STEADY else [1, 0] for _ in xrange(n_timepoints - 1)]
                        for dyn in self.dynamics]
        self.n_dynamics_params = 2

        # array for the prior probabilities of dynamics
        self.dynamic_priors = [[1. / 3] * (n_timepoints - 1) for _ in xrange(self.n_dynamics)]


    def init_caches(self, blocks):
        max_distance = max(len(block[FOREGROUND_SIGNAL][0]) + 2 for block in blocks)
        # echo('Initializing caches for max_distance:', max_distance)
        clib.init_caches(self.n_timepoints - 1,
                         max_distance,
                         self.n_dynamics)

    def free_caches(self):
        clib.free_caches(self.n_timepoints - 1, self.n_dynamics)


    def init_from_file(self, filename):

        echo('Loading model from:', filename)
        # with open(filename) as model_f:
        #     self.__dict__ = pickle.load(model_f)

        with open(filename) as model_f:
            for key, value in pickle.load(model_f).items():
                self.__dict__[key] = value

        # self.print_model()

    def save_model(self):
        with open(self.model_fname, 'w') as model_f:
            pickle.dump(self.__dict__, model_f)

    def print_model(self):
        to_print = '\nForeground:\nDeltas\n' + '\t'.join(['%.10lf' % v for v in self.foreground_delta]) + '\n'
        to_print += 'Betas\n' + '\n'.join(['\t'.join('%.10lf' % self.foreground_betas[t][i] for t in xrange(self.n_timepoints)) for i in xrange(self.n_covariates)]) + '\n'
        to_print += '\nBackground:\nDeltas\n' + '\t'.join(['%.10lf' % v for v in self.background_delta]) + '\n'
        to_print += 'Betas\n' + '\n'.join(['\t'.join('%.10lf' % self.background_betas[t][i] for t in xrange(self.n_timepoints)) for i in xrange(self.n_covariates)]) + '\n'

        for dyn_idx, dyn in enumerate(self.dynamics):
            to_print += 'Dynamic:' + dyn + '\n'
            to_print += 'Params:' + '\t' + '\t'.join(', '.join('%.10lf' % vv for vv in v) for v in self.dynamics_params[dyn_idx]) + '\n'
            to_print += 'Priors:' + '\t' + '\t'.join('%.10lf' % v for v in self.dynamic_priors[dyn_idx]) + '\n'

        to_print += 'FDR for decoding:\t' + '\t'.join(map(str, self.fdr_threshold_for_decoding)) + '\n'

        echo(to_print, level=ECHO_TO_LOGFILE)

    def boundary_movement_model(self, diff, timepoint, return_log=False):

        if diff == 0:
            move_prob = 0 if return_log else 1
            dyn = self.dynamics_idx[STEADY]
        else:
            if diff > 0:
                dyn = self.dynamics_idx[EXPAND]
            else:
                dyn = self.dynamics_idx[CONTRACT]

            move_prob = self.nb_model(abs(diff) - 1, [1],
                                      [self.dynamics_params[dyn][timepoint][1]],
                                       self.dynamics_params[dyn][timepoint][0])

        if return_log:
            if self.dynamic_priors[dyn][timepoint] > 0:
                log_prior = math.log(self.dynamic_priors[dyn][timepoint])
            else:
                log_prior = -float('inf')
            return move_prob + log_prior
        else:
            return move_prob * self.dynamic_priors[dyn][timepoint]

    def foreground_model(self, timepoint_idx, signal, covariates):
        return self.nb_model(signal,
                             covariates,
                             self.foreground_betas[timepoint_idx],
                             self.foreground_delta[timepoint_idx])

    def background_model(self, timepoint_idx, signal, covariates):
        return self.nb_model(signal,
                             covariates,
                             self.background_betas[timepoint_idx],
                             self.background_delta[timepoint_idx])

    def nb_model(self, signal, covariates, betas, delta):
        nb_mean = math.exp(sum(c * beta for c, beta in izip(covariates, betas)))

        return (math.lgamma(signal + delta) - math.lgamma(signal + 1) - math.lgamma(delta) +
                delta * (math.log(delta / (nb_mean + delta))) +
                signal * (math.log(nb_mean / (nb_mean + delta)))
                )

    def calculate_signal_cache(self, block_fgr, covariates):

        block_length = len(block_fgr[0])

        n_timepoints = self.n_timepoints

        emission_cache = cube(n_timepoints, block_length + 1, block_length + 1, default=-float('inf'))
        # return emission_cache

        for t_idx in xrange(n_timepoints):
            cur_cache = emission_cache[t_idx]

            t_fgr_signal = block_fgr[t_idx]

            all_background_log_prob = sum(self.background_model(t_idx, fgr, covariates[t_idx][pos_idx])
                                          for pos_idx, fgr in enumerate(t_fgr_signal))

            left_flanking_log_prob = 0

            for start in xrange(block_length + 1):
                peak_log_prob = 0

                if start > 0:
                    left_flanking_log_prob += self.background_model(t_idx, t_fgr_signal[start - 1], covariates[t_idx][start - 1])

                right_flanking_log_prob = all_background_log_prob - left_flanking_log_prob

                for end in xrange(start, block_length + 1):

                    if start == end:
                        cur_cache[start][end] = all_background_log_prob
                        continue

                    peak_log_prob += self.foreground_model(t_idx, t_fgr_signal[end - 1], covariates[t_idx][end - 1])

                    right_flanking_log_prob -= self.background_model(t_idx, t_fgr_signal[end - 1], covariates[t_idx][end - 1])

                    cur_cache[start][end] = left_flanking_log_prob + peak_log_prob + right_flanking_log_prob

        return emission_cache # left_flanking_reads, right_flanking_reads

    def forward(self, emission_cache, n_timepoints, block_length):

        result = [matrix(block_length + 1, block_length + 1, default=-float('inf')) for _ in xrange(n_timepoints)]
        result[0] = matcopy(emission_cache[0])

        for t in xrange(1, n_timepoints):

            for cur_start in xrange(block_length + 1):

                for cur_end in xrange(cur_start, block_length + 1):

                    gamma = -float('inf')

                    for prev_start in xrange(block_length + 1):
                        for prev_end in xrange(prev_start, block_length + 1):

                            log_prob = result[t - 1][prev_start][prev_end] + \
                                       self.boundary_movement_model(prev_start - cur_start, t - 1, return_log=True) + \
                                       self.boundary_movement_model(cur_end - prev_end, t - 1, return_log=True)

                            gamma = add_log_probs(gamma, log_prob)

                    gamma += emission_cache[t][cur_start][cur_end]
                    result[t][cur_start][cur_end] = gamma

        return result

    def backward(self, emission_cache, n_timepoints, block_length):

        result = [matrix(block_length + 1, block_length + 1, default=-float('inf')) for _ in xrange(n_timepoints)]
        for cur_start in xrange(block_length + 1):
            for cur_end in xrange(cur_start, block_length + 1):
                result[-1][cur_start][cur_end] = 0

        for t in reversed(xrange(n_timepoints - 1)):

            for cur_start in xrange(block_length + 1):

                for cur_end in xrange(cur_start, block_length + 1):

                    gamma = -float('inf')
                    for next_start in xrange(block_length + 1):
                        for next_end in xrange(next_start, block_length + 1):
                            log_prob = result[t + 1][next_start][next_end] + \
                                       emission_cache[t + 1][next_start][next_end] + \
                                       self.boundary_movement_model(cur_start - next_start, t, return_log=True) + \
                                       self.boundary_movement_model(next_end - cur_end, t, return_log=True)

                            gamma = add_log_probs(gamma, log_prob)

                    result[t][cur_start][cur_end] = gamma

        return result

    def dist_to_dynamic_idx(self, dist):
        return (self.dynamics_idx[STEADY] if dist == 0
                else self.dynamics_idx[EXPAND] if dist > 0
                else self.dynamics_idx[CONTRACT])

    def EM_step(self, block, param_info):

        n_timepoints = self.n_timepoints
        block_fgr = block[FOREGROUND_SIGNAL]

        block_length = len(block_fgr[0])

        emission_cache = self.calculate_signal_cache(block_fgr, block[BLOCK_COVARIATES])

        # emission_cache, all_expected_reads_per_bin = \
        #     self.calculate_signal_cache(block_fgr, block_bgr, total_fgr_reads, total_bgr_reads)

        F = self.forward(emission_cache, n_timepoints, block_length)
        B = self.backward(emission_cache, n_timepoints, block_length)

        # rescale cluster posteriors to sum to one
        log_likelihood = reduce(add_log_probs,
                                sorted(p for r in F[-1] for p in r if not (math.isnan(p) or math.isinf(p))),
                                -float('inf'))

        position_posteriors = matrix(block_length + 1, block_length + 1, default=-float('inf'))

        # add the current cluster posterior to the total cluster posterior
        peak_posteriors = matrix(n_timepoints, block_length, default=0)
        for t in xrange(n_timepoints - 1):
            set_matrix(position_posteriors, -float('inf'))
            # update_info = []
            dynamics_posterior = [[-float('inf')] * self.n_dynamics for _ in xrange(2)]

            dist_posteriors = [[[-float('inf')] * (block_length + 1) for _ in xrange(self.n_dynamics)] for _ in xrange(2)]

            for cur_start in xrange(block_length + 1):

                for cur_end in xrange(cur_start, block_length + 1):

                    for next_start in xrange(block_length + 1):
                        start_dist = cur_start - next_start

                        for next_end in xrange(next_start, block_length + 1):
                            end_dist = next_end - cur_end

                            log_prob = F[t][cur_start][cur_end] + \
                                       self.boundary_movement_model(start_dist, t, return_log=True) + \
                                       self.boundary_movement_model(end_dist, t, return_log=True) + \
                                       B[t + 1][next_start][next_end] + \
                                       emission_cache[t + 1][next_start][next_end]

                            position_posteriors[cur_start][cur_end] = add_log_probs(position_posteriors[cur_start][cur_end],
                                                                                    log_prob)

                            # update_info.append((cur_start, cur_end, start_dist, end_dist, log_prob))

                            l_dyn = self.dist_to_dynamic_idx(start_dist)
                            r_dyn = self.dist_to_dynamic_idx(end_dist)

                            for boundary_idx, (dyn, dist) in enumerate([(l_dyn, start_dist),
                                                                        (r_dyn, end_dist )]):

                                dist_posteriors[boundary_idx][dyn][abs(dist)] = add_log_probs(dist_posteriors[boundary_idx][dyn][abs(dist)],
                                                                                              log_prob)

                                dynamics_posterior[boundary_idx][dyn] = add_log_probs(dynamics_posterior[boundary_idx][dyn],
                                                                                      log_prob)

            for boundary_idx in xrange(2):
                dynamics_posterior[boundary_idx] = convert_and_normalize_log_posteriors(dynamics_posterior[boundary_idx])

                # store the posteriors for each dynamic on each side
                for dyn_idx in xrange(self.n_dynamics):
                    param_info[TOTAL_POSTERIORS_PER_DYNAMIC][dyn_idx][t] += dynamics_posterior[boundary_idx][dyn_idx]

                    dyn_dist_posteriors = convert_and_normalize_log_posteriors(dist_posteriors[boundary_idx][dyn_idx])

                    if dyn_idx > 0:

                        for dist in xrange(1, len(dyn_dist_posteriors)):

                            weight = dyn_dist_posteriors[dist] * dynamics_posterior[boundary_idx][dyn_idx]

                            param_info[DYNAMICS_PARAMS][dyn_idx][t][LAMBDA] += (1 if dyn_idx == self.dynamics_idx[EXPAND] else -1) * dist * weight
                            param_info[DYNAMICS_PARAMS][dyn_idx][t][SCALE] += weight

            convert_and_normalize_log_matrix(position_posteriors, math.e)

            for start in xrange(block_length + 1):
                for end in xrange(start, block_length + 1):

                    weight = position_posteriors[start][end]

                    for pos in xrange(start, end):
                        peak_posteriors[t][pos] += weight

                        # param_info[EMISSION_PARAMS][t][FOREGROUND_SIGNAL][MEAN] += weight * block_fgr[t][pos]
                        # param_info[EMISSION_PARAMS][t][FOREGROUND_SIGNAL][EXPECTED_MEAN] += weight * all_expected_reads_per_bin[t]

        # update the emission parameters for the last time point
        t = n_timepoints - 1
        position_posteriors = matcopy(F[-1])
        convert_and_normalize_log_matrix(position_posteriors, math.e)

        for start in xrange(block_length + 1):
            for end in xrange(start, block_length + 1):

                weight = position_posteriors[start][end]

                for pos in xrange(start, end):
                    peak_posteriors[t][pos] += weight

                    # param_info[EMISSION_PARAMS][t][FOREGROUND_SIGNAL][MEAN] += weight * block_fgr[t][pos]
                    # param_info[EMISSION_PARAMS][t][FOREGROUND_SIGNAL][EXPECTED_MEAN] += weight * \
                    #                                                                     all_expected_reads_per_bin[t]

        param_info[PEAK_POSTERIORS][block[BLOCK_ID]] = peak_posteriors
        return log_likelihood

    def get_model_parameters_as_C_types(self, block):

        n_timepoints = self.n_timepoints
        n_dynamics = self.n_dynamics
        n_covariates = self.n_covariates

        block_length = len(block[FOREGROUND_SIGNAL][0])

        C_block_fgr = (ctypes.c_int * (n_timepoints * block_length))(*[block[FOREGROUND_SIGNAL][t][p]
                                                                         for t in xrange(n_timepoints)
                                                                         for p in xrange(block_length)])

        C_block_covariates = (C_DOUBLE * (n_timepoints * block_length * n_covariates))(*[
            block[BLOCK_COVARIATES][t][p][c]
                for t in xrange(n_timepoints)
                    for p in xrange(block_length)
                        for c in xrange(n_covariates)
        ])

        C_foreground_delta = (C_DOUBLE * n_timepoints)(*self.foreground_delta)
        C_foreground_betas = (C_DOUBLE * (n_timepoints * n_covariates))(*[b for t in xrange(n_timepoints) for b in self.foreground_betas[t]])

        C_background_delta = (C_DOUBLE * n_timepoints)(*self.background_delta)
        C_background_betas = (C_DOUBLE * (n_timepoints * n_covariates))(*[b for t in xrange(n_timepoints) for b in self.background_betas[t]])

        # the priors array
        C_priors = (C_DOUBLE * (n_dynamics * (n_timepoints - 1)))\
            (*[self.dynamic_priors[d_idx][t_idx]
               for d_idx in xrange(n_dynamics)
                for t_idx in xrange(n_timepoints - 1)])

        # C_lambdas = (C_DOUBLE * (self.n_dynamics * 2 * (n_timepoints - 1))) \
        #                                 (*[l for dynamic_lambdas in self.lambdas for l in dynamic_lambdas])

        C_dynamics_params = (C_DOUBLE * (self.n_dynamics * self.n_dynamics_params * (n_timepoints - 1))) \
                                        (*[self.dynamics_params[dyn][t][p_idx]
                                           for dyn in xrange(self.n_dynamics)
                                                for t in xrange(self.n_timepoints - 1)
                                                    for p_idx in xrange(self.n_dynamics_params)])

        return ( C_block_fgr,
                 C_block_covariates,
                 block_length,

                 C_foreground_delta,
                 C_foreground_betas,

                 C_background_delta,
                 C_background_betas,

                 C_dynamics_params,
                 n_dynamics,
                 C_priors)


    def EM_step_C(self, block, param_info):

        C_log_likelihood = (C_DOUBLE * 1)(0.)

        (C_block_fgr,
         C_block_covariates,
         C_block_length,

         C_foreground_delta,
         C_foreground_betas,

         C_background_delta,
         C_background_betas,

         C_dynamics_params,
         C_n_dynamics,
         C_priors) = self.get_model_parameters_as_C_types(block)

        # construct an empty param_array
        n_timepoints = self.n_timepoints
        C_param_array = (C_DOUBLE * (C_block_length * n_timepoints + # space for peak posteriors for each position
                                     self.n_dynamics * (n_timepoints - 1) + # space for total posteriors per dynamic
                                     # (self.n_dynamics_params + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
                                     (C_block_length + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
                                     ))()

        # print observed[LEFT_BOUNDARY], observed[RIGHT_BOUNDARY]
        status = clib.EM_step(C_block_fgr,
                              C_block_covariates,
                              C_block_length,

                              self.n_covariates,
                              self.n_timepoints,

                              C_foreground_delta,
                              C_foreground_betas,

                              C_background_delta,
                              C_background_betas,

                              C_dynamics_params,
                              C_n_dynamics,
                              self.n_dynamics_params,

                              C_priors,

                              C_param_array,

                              C_log_likelihood)

        if status == 1:
            raise UnderflowException

        # print 'param_array:'
        # print param_array[:len(param_array)]
        # copy results from param_array to param_info
        self.param_info_from_array(param_info, C_param_array, block[BLOCK_ID], C_block_length)

        # update signal delta params in param_info
        # print list(bidirectional_cluster_posteriors)

        # print pprint.pformat(param_info)
        # print 'total likelihood:',total_log_likelihood.value

        return C_log_likelihood[0]


    def EM_step_split_on_positions_C(self, block, param_info):

        if block[BLOCK_LENGTH] > self.max_region_length:
            split_positions = block[SPLIT_POINT]
        else:
            split_positions = xrange(block[BLOCK_LENGTH] + 1)

        log_likelihood = -float('inf')

        split_log_likelihoods = []
        split_param_arrays = [[] for _ in split_positions]

        for split_idx, split_point in enumerate(split_positions):
            split_log_likelihood = 0

            for side, side_start, side_end, side_offset in [(LEFT_BOUNDARY, 0, split_point, split_point),
                                               (RIGHT_BOUNDARY, split_point, block[BLOCK_LENGTH], split_point)]:

                split_block = self.split_block(block, side, side_start, side_end, side_offset)

                if split_block[BLOCK_LENGTH] > 0:
                    (C_block_fgr,
                     C_block_covariates,
                     C_block_length,

                     C_foreground_delta,
                     C_foreground_betas,

                     C_background_delta,
                     C_background_betas,

                     C_dynamics_params,
                     C_n_dynamics,
                     C_priors) = self.get_model_parameters_as_C_types(split_block)

                    C_log_likelihood = (C_DOUBLE * 1)(0.)

                    # construct an empty param_array
                    n_timepoints = self.n_timepoints
                    C_param_array = (C_DOUBLE * (C_block_length * n_timepoints + # space for peak posteriors for each position
                                                 self.n_dynamics * (n_timepoints - 1) + # space for total posteriors per dynamic
                                                 (C_block_length + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
                                                 # (self.n_dynamics_params + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
                                                 ))()

                    # print observed[LEFT_BOUNDARY], observed[RIGHT_BOUNDARY]
                    status = clib.EM_step_split(C_block_fgr,
                                                C_block_covariates,
                                                C_block_length,

                                                self.n_covariates,
                                                self.n_timepoints,

                                                C_foreground_delta,
                                                C_foreground_betas,

                                                C_background_delta,
                                                C_background_betas,

                                                C_dynamics_params,
                                                C_n_dynamics,
                                                self.n_dynamics_params,

                                                C_priors,

                                                C_param_array,

                                                C_log_likelihood)

                    if status == 1:
                        raise UnderflowException

                    # copy results from param_array to param_info
                    split_param_arrays[split_idx].append((C_param_array,
                                                            block[BLOCK_ID],
                                                            C_block_length,
                                                            side,
                                                            side_start))

                    split_log_likelihood += C_log_likelihood[0]
                else:
                    split_log_likelihood += sum(math.log(steady_prior) for steady_prior in self.dynamic_priors[0])
                    fake_param_array = [1 if dynamics_idx == 0 else 0 for dynamics_idx in xrange(self.n_dynamics)
                                        for timepoint_idx in xrange(self.n_timepoints - 1)] + \
                                       [0 for dyn_idx in xrange(self.n_dynamics)
                                        for t in xrange(self.n_timepoints - 1)]

                    split_param_arrays[split_idx].append((fake_param_array,
                                                            block[BLOCK_ID],
                                                            0,
                                                            side,
                                                            side_start))

            split_log_likelihoods.append(split_log_likelihood)

            log_likelihood = add_log_probs(log_likelihood, split_log_likelihood)

        split_weights = convert_and_normalize_log_posteriors(split_log_likelihoods)

        for split_idx, weight in enumerate(split_weights):
            for (C_param_array, block_id, C_block_length, side, side_start) in split_param_arrays[split_idx]:
                self.param_info_from_array_split(param_info,
                                                 C_param_array,
                                                 block_id,
                                                 C_block_length,
                                                 side,
                                                 side_start,
                                                 factor=weight)


        return log_likelihood

    def EM_step_split_C(self, block, param_info):

        split_positions = [block[SPLIT_POINT][0]]

        log_likelihood = -float('inf')

        split_log_likelihoods = []
        split_param_arrays = [[] for _ in split_positions]

        for split_idx, split_point in enumerate(split_positions):
            split_log_likelihood = 0

            for side, side_start, side_end, side_offset in [(LEFT_BOUNDARY, 0, split_point, split_point),
                                                            (RIGHT_BOUNDARY, split_point, block[BLOCK_LENGTH], split_point)]:

                split_block = self.split_block(block, side, side_start, side_end, side_offset)

                if split_block[BLOCK_LENGTH] > 0:
                    (C_block_fgr,
                     C_block_covariates,
                     C_block_length,

                     C_foreground_delta,
                     C_foreground_betas,

                     C_background_delta,
                     C_background_betas,

                     C_dynamics_params,
                     C_n_dynamics,
                     C_priors) = self.get_model_parameters_as_C_types(split_block)

                    C_log_likelihood = (C_DOUBLE * 1)(0.)

                    # construct an empty param_array
                    n_timepoints = self.n_timepoints
                    C_param_array = (C_DOUBLE * (C_block_length * n_timepoints + # space for peak posteriors for each position
                                                 self.n_dynamics * (n_timepoints - 1) + # space for total posteriors per dynamic
                                                 (C_block_length + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
                                                 # (self.n_dynamics_params + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
                                                 ))()

                    # print observed[LEFT_BOUNDARY], observed[RIGHT_BOUNDARY]
                    status = clib.EM_step_split(C_block_fgr,
                                                C_block_covariates,
                                                C_block_length,

                                                self.n_covariates,
                                                self.n_timepoints,

                                                C_foreground_delta,
                                                C_foreground_betas,

                                                C_background_delta,
                                                C_background_betas,

                                                C_dynamics_params,
                                                C_n_dynamics,
                                                self.n_dynamics_params,

                                                C_priors,

                                                C_param_array,

                                                C_log_likelihood)

                    if status == 1:
                        raise UnderflowException

                    # copy results from param_array to param_info
                    split_param_arrays[split_idx].append((C_param_array,
                                                            block[BLOCK_ID],
                                                            C_block_length,
                                                            side,
                                                            side_start))

                    split_log_likelihood += C_log_likelihood[0]
                else:
                    split_log_likelihood += sum(math.log(steady_prior) for steady_prior in self.dynamic_priors[0])
                    fake_param_array = [1 if dynamics_idx == 0 else 0 for dynamics_idx in xrange(self.n_dynamics)
                                        for timepoint_idx in xrange(self.n_timepoints - 1)] + \
                                       [0 for dyn_idx in xrange(self.n_dynamics)
                                        for t in xrange(self.n_timepoints - 1)]

                    split_param_arrays[split_idx].append((fake_param_array,
                                                            block[BLOCK_ID],
                                                            0,
                                                            side,
                                                            side_start))

            split_log_likelihoods.append(split_log_likelihood)

            log_likelihood = add_log_probs(log_likelihood, split_log_likelihood)

        split_weights = convert_and_normalize_log_posteriors(split_log_likelihoods)

        for split_idx, weight in enumerate(split_weights):
            for (C_param_array, block_id, C_block_length, side, side_start) in split_param_arrays[split_idx]:
                self.param_info_from_array_split(param_info,
                                                 C_param_array,
                                                 block_id,
                                                 C_block_length,
                                                 side,
                                                 side_start,
                                                 factor=weight)

        return log_likelihood

    # def EM_step_split_on_every_position_C(self, block, param_info):
    #
    #     log_likelihood = -float('inf')
    #
    #     split_log_likelihoods = []
    #     split_param_arrays = [[] for _ in xrange(block[BLOCK_LENGTH] - 1)]
    #
    #     for split_idx, split_point in enumerate(xrange(1, block[BLOCK_LENGTH])):
    #         split_log_likelihood = 0
    #
    #         for side, side_start, side_end in [(LEFT_BOUNDARY, 0, split_point),
    #                                            (RIGHT_BOUNDARY, split_point, block[BLOCK_LENGTH])]:
    #
    #             split_block = self.split_block(block, side, side_start, side_end)
    #
    #             if split_block[BLOCK_LENGTH] > 0:
    #                 (C_block_fgr,
    #                  C_block_covariates,
    #                  C_block_length,
    #
    #                  C_foreground_delta,
    #                  C_foreground_betas,
    #
    #                  C_background_delta,
    #                  C_background_betas,
    #
    #                  C_dynamics_params,
    #                  C_n_dynamics,
    #                  C_priors) = self.get_model_parameters_as_C_types(split_block)
    #
    #                 C_log_likelihood = (C_DOUBLE * 1)(0.)
    #
    #                 # construct an empty param_array
    #                 n_timepoints = self.n_timepoints
    #                 C_param_array = (C_DOUBLE * (C_block_length * n_timepoints + # space for peak posteriors for each position
    #                                              self.n_dynamics * (n_timepoints - 1) + # space for total posteriors per dynamic
    #                                              (C_block_length + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
    #                                              # (self.n_dynamics_params + 1) * self.n_dynamics * (n_timepoints - 1) # space for lambdas parameters
    #                                              ))()
    #
    #                 # print observed[LEFT_BOUNDARY], observed[RIGHT_BOUNDARY]
    #                 status = clib.EM_step_split(C_block_fgr,
    #                                             C_block_covariates,
    #                                               C_block_length,
    #
    #                                               self.n_covariates,
    #                                               self.n_timepoints,
    #
    #                                               C_foreground_delta,
    #                                               C_foreground_betas,
    #
    #                                               C_background_delta,
    #                                               C_background_betas,
    #
    #                                               C_dynamics_params,
    #                                               C_n_dynamics,
    #                                               self.n_dynamics_params,
    #
    #
    #                                               C_priors,
    #
    #                                               C_param_array,
    #
    #                                               C_log_likelihood)
    #
    #                 if status == 1:
    #                     raise UnderflowException
    #
    #                 # copy results from param_array to param_info
    #                 split_param_arrays[split_idx].append((C_param_array,
    #                                                       block[BLOCK_ID],
    #                                                       C_block_length,
    #                                                       side,
    #                                                       side_start))
    #
    #                 # self.param_info_from_array_split(split_param_info,
    #                 #                                  C_param_array,
    #                 #                                  block[BLOCK_ID],
    #                 #                                  C_block_length,
    #                 #                                  side,
    #                 #                                  side_start,
    #                 #                                  factor = 1)
    #
    #                 split_log_likelihood += C_log_likelihood[0]
    #
    #         split_log_likelihoods.append(split_log_likelihood)
    #
    #         log_likelihood = add_log_probs(log_likelihood, split_log_likelihood)
    #
    #     split_weights = convert_and_normalize_log_posteriors(split_log_likelihoods)
    #
    #     for split_idx, weight in enumerate(split_weights):
    #         for (C_param_array, block_id, C_block_length, side, side_start) in split_param_arrays[split_idx]:
    #             self.param_info_from_array_split(param_info,
    #                                              C_param_array,
    #                                              block_id,
    #                                              C_block_length,
    #                                              side,
    #                                              side_start,
    #                                              factor=weight)
    #
    #     return log_likelihood

    def new_param_info(self, block=None):

        return { DYNAMICS_PARAMS: [[[0.] * (self.n_dynamics_params + 1) for _ in xrange(self.n_timepoints - 1)]
                                   for _ in xrange(self.n_dynamics)],

                 EMISSION_PARAMS: [{FOREGROUND_SIGNAL: {MEAN: 0, EXPECTED_MEAN: 0}}
                                    for _ in xrange(self.n_timepoints)],

                 TOTAL_POSTERIORS_PER_DYNAMIC: matrix(self.n_dynamics, self.n_timepoints - 1),
                 PEAK_POSTERIORS: {block[BLOCK_ID]: matrix(self.n_timepoints, block[BLOCK_LENGTH])} if block is not None else {},
                 DYNAMICS_JUMP_POSTERIORS: {block[BLOCK_ID]: cube(self.n_dynamics, self.n_timepoints - 1, block[BLOCK_LENGTH] + 1)} if block is not None else {}
                }

    def add_param_info(self, from_info, to_info):
        for d in xrange(self.n_dynamics):
            for t in xrange(self.n_timepoints - 1):
                for param_idx in xrange(self.n_dynamics_params + 1):
                    to_info[DYNAMICS_PARAMS][d][t][param_idx] += from_info[DYNAMICS_PARAMS][d][t][param_idx]

            for t in xrange(self.n_timepoints - 1):
                to_info[TOTAL_POSTERIORS_PER_DYNAMIC][d][t] += from_info[TOTAL_POSTERIORS_PER_DYNAMIC][d][t]

        for t in xrange(self.n_timepoints):
            for signal_type in [FOREGROUND_SIGNAL]:
                for key in [MEAN, EXPECTED_MEAN]:
                    to_info[EMISSION_PARAMS][t][signal_type][key] += from_info[EMISSION_PARAMS][t][signal_type][key]

        for key in from_info[PEAK_POSTERIORS]:
            to_info[PEAK_POSTERIORS][key] = from_info[PEAK_POSTERIORS][key]

        for key in from_info[DYNAMICS_JUMP_POSTERIORS]:
            to_info[DYNAMICS_JUMP_POSTERIORS][key] = from_info[DYNAMICS_JUMP_POSTERIORS][key]

    def clear_param_info(self, info):
        for d in xrange(self.n_dynamics):
            for t in xrange(self.n_timepoints - 1):
                for param_idx in xrange(self.n_dynamics_params + 1):
                    info[DYNAMICS_PARAMS][d][t][param_idx] = 0

            for t in xrange(self.n_timepoints - 1):
                info[TOTAL_POSTERIORS_PER_DYNAMIC][d][t] = 0

        for t in xrange(self.n_timepoints):
            for signal_type in [FOREGROUND_SIGNAL]:
                for key in [MEAN, EXPECTED_MEAN]:
                    info[EMISSION_PARAMS][t][signal_type][key] = 0
        info[PEAK_POSTERIORS] = {}
        info[DYNAMICS_JUMP_POSTERIORS] = {}

    def param_info_from_array(self, param_info, param_array, block_id, block_length):

        # peak_posteriors = matrix(self.n_timepoints, block_length)
        # copy the noise parameters
        for t in xrange(self.n_timepoints):
            for p in xrange(block_length):
                param_info[PEAK_POSTERIORS][block_id][t][p] = param_array[t * block_length + p]

        # param_info[PEAK_POSTERIORS][block_id] = peak_posteriors

        # copy the expand parameters
        offset = block_length * self.n_timepoints
        for dynamics_idx in xrange(self.n_dynamics):
            for timepoint_idx in xrange(self.n_timepoints - 1):
                param_info[TOTAL_POSTERIORS_PER_DYNAMIC][dynamics_idx][timepoint_idx] = \
                    param_array[offset + dynamics_idx * (self.n_timepoints - 1) + timepoint_idx]

        offset += self.n_dynamics * (self.n_timepoints - 1)

        # dynamics_jump_posteriors = cube(self.n_dynamics, self.n_timepoints - 1, block_length + 1)
        # copy the noise parameters
        for dyn_idx in xrange(1, self.n_dynamics):
            for t in xrange(self.n_timepoints - 1):
                djp_offset = offset + dyn_idx * (block_length + 1) * (self.n_timepoints - 1) + t * (block_length + 1)

                for p in xrange(block_length + 1):
                    param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dyn_idx][t][p] = param_array[djp_offset + p]

        # param_info[DYNAMICS_JUMP_POSTERIORS][block_id] = dynamics_jump_posteriors

        return param_info

    def param_info_from_array_split(self, param_info, param_array, block_id, block_length, side, side_start, factor=1.):

        peak_posteriors = [0] * block_length

        # copy the noise parameters
        for t in xrange(self.n_timepoints):
            for p in xrange(block_length):
                peak_posteriors[p] = param_array[t * block_length + p]

            if side == LEFT_BOUNDARY:
                peak_posteriors = list(reversed(peak_posteriors))

            for p in xrange(block_length):
                param_info[PEAK_POSTERIORS][block_id][t][side_start + p] += peak_posteriors[p] * factor

        # copy the expand parameters
        offset = block_length * self.n_timepoints
        for dynamics_idx in xrange(self.n_dynamics):
            for timepoint_idx in xrange(self.n_timepoints - 1):
                param_info[TOTAL_POSTERIORS_PER_DYNAMIC][dynamics_idx][timepoint_idx] += \
                    param_array[offset + dynamics_idx * (self.n_timepoints - 1) + timepoint_idx] * factor

        offset += self.n_dynamics * (self.n_timepoints - 1)

        # copy the jump posteriors
        for dyn_idx in xrange(1, self.n_dynamics):
            for t in xrange(self.n_timepoints - 1):
                djp_offset = offset + dyn_idx * (block_length + 1) * (self.n_timepoints - 1) + t * (block_length + 1)

                for p in xrange(block_length + 1):
                    param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dyn_idx][t][p] += param_array[djp_offset + p] * factor

        return param_info

    # def param_info_from_array_split(self, param_info, param_array, block_id, block_length, side):
    #
    #     peak_posteriors = matrix(self.n_timepoints, block_length)
    #     # copy the noise parameters
    #     for t in xrange(self.n_timepoints):
    #         for p in xrange(block_length):
    #             peak_posteriors[t][p] = param_array[t * block_length + p]
    #
    #     if side == LEFT_BOUNDARY:
    #         flip = lambda a: list(reversed(a))
    #     else:
    #         flip = lambda a: a
    #
    #     if block_id not in param_info[PEAK_POSTERIORS]:
    #         param_info[PEAK_POSTERIORS][block_id] = [[] for _ in xrange(self.n_timepoints)]
    #
    #     for t in xrange(self.n_timepoints):
    #         param_info[PEAK_POSTERIORS][block_id][t].extend(flip(peak_posteriors[t]))
    #
    #     # copy the expand parameters
    #     offset = block_length * self.n_timepoints
    #     for dynamics_idx in xrange(self.n_dynamics):
    #         for timepoint_idx in xrange(self.n_timepoints - 1):
    #             param_info[TOTAL_POSTERIORS_PER_DYNAMIC][dynamics_idx][timepoint_idx] += \
    #                 param_array[offset + dynamics_idx * (self.n_timepoints - 1) + timepoint_idx]
    #
    #     offset += self.n_dynamics * (self.n_timepoints - 1)
    #
    #     dynamics_jump_posteriors = cube(self.n_dynamics, self.n_timepoints - 1, block_length + 1)
    #
    #     # copy the noise parameters
    #     for dyn_idx in xrange(1, self.n_dynamics):
    #         for t in xrange(self.n_timepoints - 1):
    #             djp_offset = offset + dyn_idx * (block_length + 1) * (self.n_timepoints - 1) + t * (block_length + 1)
    #
    #             for p in xrange(block_length + 1):
    #                 dynamics_jump_posteriors[dyn_idx][t][p] = param_array[djp_offset + p]
    #
    #     if block_id not in param_info[DYNAMICS_JUMP_POSTERIORS]:
    #         param_info[DYNAMICS_JUMP_POSTERIORS][block_id] = [[[] for _ in xrange(self.n_timepoints - 1)]
    #                                                           for _ in xrange(self.n_dynamics)]
    #
    #     for dyn_idx in xrange(self.n_dynamics):
    #         for t in xrange(self.n_timepoints - 1):
    #             if len(param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dyn_idx][t]) < block_length + 1:
    #                 diff_length = block_length + 1 - len(param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dyn_idx][t])
    #                 param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dyn_idx][t].extend([0] * diff_length)
    #
    #             for p in xrange(block_length + 1):
    #                 param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dyn_idx][t][p] += dynamics_jump_posteriors[dyn_idx][t][p]
    #
    #     return param_info

    def compute_FDR_threshold(self, blocks, fdr=0.01):
        echo('Computing FDR threshold')
        # self.n_threads = 1
        if self.n_threads > 1:
            pool = Pool(processes=self.n_threads)
            _map = pool.map
        else:
            _map = map

        CHUNK_SIZE = max(1, min(500, 1 + len(blocks) / self.n_threads))
        no_peak_probs = [[] for _ in xrange(self.n_timepoints)]
        for batch_no_peak_probs in _map(worker,
                                        [(compute_no_peak_probability, self, batch, batch_no)
                                            for batch_no, batch in enumerate(chunks(blocks.values(), CHUNK_SIZE))]):
            for t in xrange(self.n_timepoints):
                no_peak_probs[t].extend(batch_no_peak_probs[t])

        if self.n_threads > 1:
            pool.close()

        print >>sys.stderr, "\n"
        self.fdr_threshold_for_decoding = [None] * self.n_timepoints

        for t in xrange(self.n_timepoints):
            sorted_no_peak_probs = sorted(no_peak_probs[t])
            n = len(sorted_no_peak_probs)

            for i, p_value in enumerate(sorted_no_peak_probs):

                if float(n * p_value) / (i + 1) <= fdr:
                    self.fdr_threshold_for_decoding[t] = p_value

        echo('FDR threshold for decoding:', self.fdr_threshold_for_decoding)

    def EM(self, blocks, MIN_DELTA_LOG_LIKELIHOOD=None, echo_level=ECHO_TO_SCREEN, bruteforce_debug=False):

        n_timepoints = self.n_timepoints

        MAX_EM_ITERATIONS = 100

        if MIN_DELTA_LOG_LIKELIHOOD is None:
            MIN_DELTA_LOG_LIKELIHOOD = 10 ** -1

        delta_likelihood = 2 * MIN_DELTA_LOG_LIKELIHOOD

        EM_iteration = 0

        if self.n_threads > 1:
            pool = Pool(processes=self.n_threads)
            _map = pool.map
        else:
            _map = map

        while EM_iteration < MAX_EM_ITERATIONS:
            log_poisson_pmf.cache = {}

            current_total_likelihood = 0
            underflows = []

            EM_iteration += 1
            echo('EM iteration:', EM_iteration, level=echo_level)

            param_info = self.new_param_info()

            CHUNK_SIZE = max(1, min(500, 1 + len(blocks) / self.n_threads))
            echo('Total training examples=', len(blocks),
                 'batches:', 1 + len(blocks) / CHUNK_SIZE,
                 'per_batch=', CHUNK_SIZE, level=echo_level)

            for batch_param_info, batch_underflows, batch_total_likelihood in \
                _map(worker, [(EM_batch, self, batch, batch_no)
                                  for batch_no, batch in enumerate(chunks(blocks.values(),
                                                                          CHUNK_SIZE))]):

                self.add_param_info(batch_param_info, param_info)

                underflows.extend(batch_underflows)
                current_total_likelihood += batch_total_likelihood

            if not bruteforce_debug:
                print >>sys.stderr, '\n'

            # remove blocks that underflow from the training blocks
            if len(underflows) > 0:
                echo('Removing blocks that cause underflows:', underflows)
                blocks = dict((block_id, blocks[block_id]) for block_id in blocks if block_id not in underflows)
                if bruteforce_debug:
                    return

            echo('Current total log likelihood:', current_total_likelihood, level=echo_level)

            if self.prev_total_likelihood is not None:
                delta_likelihood = current_total_likelihood - self.prev_total_likelihood
                echo('EM Iteration: %d\tDelta Log Likelihood:' % EM_iteration, delta_likelihood, level=echo_level)

                if delta_likelihood < 0:
                    echo('**** ERROR!!! ****')

                    if bruteforce_debug:
                        echo('delta log likelihood=', delta_likelihood, '\tblocks:', blocks)

                    echo('+' * 100, '\n', '+' * 100)
                    raise DecreasingLikelihoodException("Decreasing likelihood. LL delta: " + str(delta_likelihood),
                                                        delta_likelihood,
                                                        blocks)

            self.prev_total_likelihood = current_total_likelihood

            # The M step
            delta = 0
            block_ids = sorted(param_info[DYNAMICS_JUMP_POSTERIORS])

            # # update the priors
            for t in xrange(n_timepoints - 1):
                total_p = sum(param_info[TOTAL_POSTERIORS_PER_DYNAMIC][d][t] for d in xrange(self.n_dynamics))
                for d in xrange(self.n_dynamics):
                    self.dynamic_priors[d][t] = param_info[TOTAL_POSTERIORS_PER_DYNAMIC][d][t] / total_p
            #
            #
            # # # update the boundary movement parameters
            for dynamic in range(1, self.n_dynamics):
                for t in range(n_timepoints - 1):
                    weights = []
                    y = []
                    scale = 0
                    for block_id in block_ids:
                        block_dynamics_jump_posteriors = param_info[DYNAMICS_JUMP_POSTERIORS][block_id][dynamic]
                        for dist, weight in enumerate(block_dynamics_jump_posteriors[t]):
                            if dist > 0:

                                y.append(dist - 1)
                                weights.append(weight)
                                scale += weight

                    if scale == 0:
                        continue

                    x = matrix(1, len(y), default=1)

                    try:
                        _delta, _betas = optimize_NegBinomial(y, x, weights, [1])
                        # _delta, _betas = optimize_NegBinomial(y, x, weights, [self.dynamics_params[dynamic][t][1]])
                        # _delta, _betas = optimize_nb_R(y, x, weights)
                    except (ZeroDeltaException, NegBinomialOptimizationFailure, PoissonOptimizationFailure):
                        echo('WARNING: An error occurred while optimizing numerically BOUNDARY MOVEMENT '
                             'NBs for time point', t, level=echo_level)
                        # do not update NB if optimization failed to find positive delta
                        continue

                    new_value = _delta
                    delta += (self.dynamics_params[dynamic][t][0] - new_value) ** 2
                    self.dynamics_params[dynamic][t][0] = new_value

                    new_value = _betas[0]
                    delta += (self.dynamics_params[dynamic][t][1] - new_value) ** 2
                    self.dynamics_params[dynamic][t][1] = new_value

            def peak_posteriors_as_np_array(peak_posteriors, timepoint_idx):
                # converts the peak posteriors to a numpy array
                return np.array([min(1, max(0, p))
                                 for block_id in block_ids
                                 for p in peak_posteriors[block_id][timepoint_idx]])

            def peak_covariates_as_np_array(blocks, timepoint_idx):
                # converts the peak posteriors to a numpy array
                return np.array([[p_covariates[cov_idx]
                                   for block_id in block_ids
                                    for p_covariates in blocks[block_id][BLOCK_COVARIATES][timepoint_idx]]
                                 for cov_idx in xrange(self.n_covariates)])

            def block_signal_as_np_array(blocks, timepoint_idx):
                # converts the peak signal to numpy array
                return np.array([s for block_id in block_ids
                                 for s in blocks[block_id][FOREGROUND_SIGNAL][timepoint_idx]])

            for (new_fgr_delta, new_fgr_betas, new_bgr_delta, new_bgr_betas, t) in _map(
                       # worker, [(optimize_a_pair_of_NegBinomials_jointly,
                         worker, [(optimize_a_pair_of_NegBinomials_jointly_shared_beta,
                                   block_signal_as_np_array(blocks, t),
                                   peak_covariates_as_np_array(blocks, t),
                                   peak_posteriors_as_np_array(param_info[PEAK_POSTERIORS], t),
                                   t,
                                   self.foreground_betas[t],
                                   self.background_betas[t],
                                   self.n_covariates)
                                    for t in xrange(self.n_timepoints)]):

                echo('time point:', t, '\tmean peak %:', mean([sum(
                    max(0, min(weight, 1.)) for weight in param_info[PEAK_POSTERIORS][block_id][t]) / len(
                    param_info[PEAK_POSTERIORS][block_id][t])
                                                 for block_id in sorted(blocks)]), '\t',
                    'mean bgr %:', mean([sum(
                    max(0, min(1 - weight, 1.)) for weight in param_info[PEAK_POSTERIORS][block_id][t]) / len(
                    param_info[PEAK_POSTERIORS][block_id][t])
                                           for block_id in sorted(blocks)])
                    , level=echo_level)

                if new_fgr_delta is not None:
                    self.foreground_delta[t] = new_fgr_delta
                    self.background_delta[t] = new_bgr_delta

                    self.foreground_betas[t] = new_fgr_betas
                    self.background_betas[t] = new_bgr_betas

            echo('New model parameters', level=echo_level)

            if not bruteforce_debug:
                self.print_model()
                self.save_model()

            echo('EM delta:', delta, level=echo_level)
            if delta_likelihood < MIN_DELTA_LOG_LIKELIHOOD:
                break

        if self.n_threads > 1:
            pool.close()

    def EM_bruteforce_debug(self, all_blocks):
        """ This method is used only for debugging purposes """

        datapoints = []
        print 'min-max block length:', min(all_blocks[b][BLOCK_LENGTH] for b in all_blocks), max(all_blocks[b][BLOCK_LENGTH] for b in all_blocks)

        for block_no, block_id in enumerate(all_blocks.keys()):
            if block_no % 100 == 0:
                echo('**** TOTAL BLOCKS:', str(len(all_blocks)), '\tBLOCKS TESTED:', block_no)
            # if all_blocks[block_id][BLOCK_LENGTH] <= self.max_region_length:
            #     continue

            self.n_timepoints = 1
            self.reset_model()
            self.max_region_length = 1000

            block = all_blocks[block_id]
            block[BLOCK_LENGTH] = 20
            # block[SPLIT_POINT] = 3
            block[FOREGROUND_SIGNAL] = [[1 for _ in xrange(block[BLOCK_LENGTH])] for _ in xrange(self.n_timepoints)]

            # block[FOREGROUND_SIGNAL] = [[random.randint(0, 10) for _ in xrange(block[BLOCK_LENGTH])],
            #                             [random.randint(0, 10) for _ in xrange(block[BLOCK_LENGTH])]]
            block[BLOCK_COVARIATES] = [[[1.0, 1.0] for _ in xrange(block[BLOCK_LENGTH])] for _ in xrange(self.n_timepoints)]

            # block = {'block_offset': 20989800, 'block_end': 20992200, 'block_id': 'chr19-1869-1', 'is_subpeak': False, 'block_length': 5, 'block_covariates': [[[1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0]]], 'split_point': 3, 'chromosome': 'chr19', 'foreground_signal': [[10, 5, 9, 6, 2], [6, 9, 0, 0, 6]]}
            # block = {'block_offset': 47914200, 'block_end': 47916400, 'block_id': 'chr19-5533-1', 'is_subpeak': False, 'block_length': 15,
            #          'block_covariates':
            #              [[[1.0] for _ in xrange(9)],
            #               # [[1.0] for _ in xrange(15)]
            #               ]
            # , 'split_point': 5, 'chromosome': 'chr19',
            #          'foreground_signal': [
            #              [0] * 2 + [0, 8, 7, 0, 8] + [0] * 2 ,
            #              # [0] * 5 + [0, 6, 10, 3, 7] + [0] * 5
            #                                ]}
            # block = {'block_offset': 36625200, 'block_end': 36627000, 'block_id': 'chr19-3831-1', 'is_subpeak': False, 'block_length': 20, 'block_covariates':
            #     [[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
            #      [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]],
            #          'split_point': 5, 'chromosome': 'chr19', 'foreground_signal':
            #     [[0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #      [6, 8, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
            try:
                # self.EM({'chr19-4329-1': {'block_offset': 40868800, 'block_end': 40871600, 'block_id': 'chr19-4329-1', 'is_subpeak': False, 'block_length': 14, 'block_covariates': [[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]], 'split_point': 5, 'chromosome': 'chr19', 'foreground_signal': [[1, 0, 3, 3, 4, 8, 8, 11, 3, 0, 4, 0, 0, 0], [1, 1, 1, 1, 4, 0, 5, 2, 1, 3, 1, 1, 1, 0], [3, 1, 2, 7, 4, 2, 3, 5, 2, 3, 2, 1, 2, 1], [1, 0, 0, 3, 3, 3, 1, 0, 5, 4, 4, 0, 0, 0], [1, 2, 1, 3, 2, 2, 1, 3, 0, 0, 0, 0, 0, 0]]}},
                # # self.EM({'chr19-2140-1': {'block_offset': 23378400, 'block_end': 23380800, 'block_id': 'chr19-2140-1', 'is_subpeak': False, 'block_length': 12, 'block_covariates': [[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]], 'split_point': 3, 'chromosome': 'chr19',
                # #                           'foreground_signal': [
                # #                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # #                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # #                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # #                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # #                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                # #                                                 ]}},
                #         MIN_DELTA_LOG_LIKELIHOOD=0.00001,
                #         echo_level=1,
                #         bruteforce_debug=False)
                self.EM({block[BLOCK_ID]: block},
                        MIN_DELTA_LOG_LIKELIHOOD=0.0000001,
                        # echo_level=1,
                        echo_level=2,
                        # bruteforce_debug=False)
                        bruteforce_debug=True)
            except DecreasingLikelihoodException as e:
                datapoints.append((e.delta_log_likelihood, e.blocks))
            # exit(1)

        print sorted(datapoints)

        exit(1)

    def compute_posteriors(self, block):
        n_timepoints = self.n_timepoints

        dynamics_posteriors = {LEFT_BOUNDARY: [{EXPAND: 0, CONTRACT: 0, STEADY: 0} for t in xrange(n_timepoints - 1)],
                               RIGHT_BOUNDARY: [{EXPAND: 0, CONTRACT: 0, STEADY: 0} for t in xrange(n_timepoints - 1)]}

        block_fgr = block[FOREGROUND_SIGNAL]

        block_length = len(block_fgr[0])

        emission_cache = self.calculate_signal_cache(block_fgr, block[BLOCK_COVARIATES])

        F = self.forward(emission_cache, n_timepoints, block_length)
        B = self.backward(emission_cache, n_timepoints, block_length)

        # add the current cluster posterior to the total cluster posterior

        for t in xrange(n_timepoints - 1):

            t_dynamics_posteriors = [[-float('inf')] * self.n_dynamics for _ in xrange(2)]

            for cur_start in xrange(block_length + 1):

                for cur_end in xrange(cur_start, block_length + 1):

                    for next_start in xrange(cur_end + 1):
                        start_dist = cur_start - next_start

                        for next_end in xrange(max(next_start, cur_start), block_length + 1):
                            end_dist = next_end - cur_end

                            log_prob = F[t][cur_start][cur_end] + \
                                       self.boundary_movement_model(start_dist, t, return_log=True) + \
                                       self.boundary_movement_model(end_dist, t, return_log=True) + \
                                       B[t + 1][next_start][next_end] + \
                                       emission_cache[t + 1][next_start][next_end]

                            l_dyn = self.dist_to_dynamic_idx(start_dist)
                            r_dyn = self.dist_to_dynamic_idx(end_dist)

                            for boundary_idx, (dyn, dist) in enumerate([(l_dyn, start_dist),
                                                                        (r_dyn, end_dist)]):

                                t_dynamics_posteriors[boundary_idx][dyn] = add_log_probs(t_dynamics_posteriors[boundary_idx][dyn],
                                                                                         log_prob)

            for boundary_idx, boundary_side in enumerate([LEFT_BOUNDARY, RIGHT_BOUNDARY]):
                cur_side_dynamics_posteriors = convert_and_normalize_log_posteriors(t_dynamics_posteriors[boundary_idx])
                for dyn_idx, dyn in enumerate([STEADY, EXPAND, CONTRACT]):
                    dynamics_posteriors[boundary_side][t][dyn] = cur_side_dynamics_posteriors[dyn_idx]

        return dynamics_posteriors

    def compute_posteriors_C(self, block):

        (C_block_fgr,
         C_block_covariates,
         C_block_length,

         C_foreground_delta,
         C_foreground_betas,

         C_background_delta,
         C_background_betas,

         C_dynamics_params,
         C_n_dynamics,
         C_priors) = self.get_model_parameters_as_C_types(block)

        C_dynamics_posteriors = (C_DOUBLE * ((self.n_timepoints - 1) * self.n_dynamics * 2))()

        status = clib.calculate_posteriors (C_block_fgr,
                                            C_block_covariates,
                                            C_block_length,

                                            self.n_covariates,
                                            self.n_timepoints,

                                            C_foreground_delta,
                                            C_foreground_betas,

                                            C_background_delta,
                                            C_background_betas,

                                            C_dynamics_params,
                                            C_n_dynamics,


                                            C_priors,

                                            C_dynamics_posteriors)

        if status == 1:
            raise UnderflowException

        dynamics_posteriors = {LEFT_BOUNDARY: [dict((k,
                                                     C_dynamics_posteriors[t * 2 * self.n_dynamics + dyn_idx])
                                                    for dyn_idx, k in enumerate([STEADY, EXPAND, CONTRACT]))
                                               for t in xrange(self.n_timepoints - 1)],

                               RIGHT_BOUNDARY: [dict((k,
                                                     C_dynamics_posteriors[t * 2 * self.n_dynamics + self.n_dynamics + dyn_idx])
                                                    for dyn_idx, k in enumerate([STEADY, EXPAND, CONTRACT]))
                                               for t in xrange(self.n_timepoints - 1)]}
        return dynamics_posteriors
    
    
    def compute_posteriors_split_C(self, block, split_point):

        dynamics_posteriors = {LEFT_BOUNDARY: [],
                               RIGHT_BOUNDARY: []}

        split_log_likelihood = 0

        for side, side_start, side_end in [(LEFT_BOUNDARY, 0, split_point),
                                           (RIGHT_BOUNDARY, split_point, block[BLOCK_LENGTH])]:

            split_block = self.split_block(block, side, side_start, side_end, split_point)

            if split_block[BLOCK_LENGTH] > 0:
                (C_block_fgr,
                 C_block_covariates,
                 C_block_length,

                 C_foreground_delta,
                 C_foreground_betas,

                 C_background_delta,
                 C_background_betas,

                 C_dynamics_params,
                 C_n_dynamics,
                 C_priors) = self.get_model_parameters_as_C_types(split_block)

                C_dynamics_posteriors = (C_DOUBLE * ((self.n_timepoints - 1) * self.n_dynamics))()
                C_total_log_likelihood = (C_DOUBLE * 1)(0)

                status = clib.calculate_posteriors_split (C_block_fgr,
                                                          C_block_covariates,
                                                          C_block_length,

                                                          self.n_covariates,
                                                          self.n_timepoints,

                                                          C_foreground_delta,
                                                          C_foreground_betas,

                                                          C_background_delta,
                                                          C_background_betas,

                                                          C_dynamics_params,
                                                          C_n_dynamics,


                                                          C_priors,

                                                          C_dynamics_posteriors,
                                                          C_total_log_likelihood)

                if status == 1:
                    raise UnderflowException

                dynamics_posteriors[side] = [dict((k, C_dynamics_posteriors[t * self.n_dynamics + dyn_idx])
                                                        for dyn_idx, k in enumerate([STEADY, EXPAND, CONTRACT]))
                                                            for t in xrange(self.n_timepoints - 1)]

                split_log_likelihood += C_total_log_likelihood[0]
            else:

                split_log_likelihood += sum(math.log(steady_prior) for steady_prior in self.dynamic_priors[0])

                dynamics_posteriors[side] = [dict((k, 1 if k == STEADY else 0)
                                                  for dyn_idx, k in enumerate([STEADY, EXPAND, CONTRACT]))
                                             for t in xrange(self.n_timepoints - 1)]

        return split_log_likelihood, dynamics_posteriors

    def compute_posteriors_split_on_positions_C(self, block):

        split_dynamics_posteriors = []
        split_likelihoods = []

        dynamics_posteriors = {LEFT_BOUNDARY: [],
                               RIGHT_BOUNDARY: []}

        for split_point in block[SPLIT_POINT]:
            cur_split_likelihood, cur_split_posteriors = self.compute_posteriors_split_C(block, split_point=split_point)
            split_likelihoods.append(cur_split_likelihood)
            split_dynamics_posteriors.append(cur_split_posteriors)

        split_posteriors = convert_and_normalize_log_posteriors(split_likelihoods)

        for side in dynamics_posteriors:
            dynamics_posteriors[side] = [dict((dynamic,
                                               sum(weigth * post[side][t][dynamic]
                                                   for weigth, post
                                                   in izip(split_posteriors, split_dynamics_posteriors)
                                                  )
                                               )
                                                for dynamic in [STEADY, EXPAND, CONTRACT])
                                            for t in xrange(self.n_timepoints - 1)]


        return dynamics_posteriors

    def compute_Viterbi_path(self, block):

        n_timepoints = self.n_timepoints
        block_fgr = block[FOREGROUND_SIGNAL]

        block_length = len(block_fgr[0])

        emission_cache = self.calculate_signal_cache(block_fgr, block[BLOCK_COVARIATES])


        DP = [matrix(block_length + 1, block_length + 1, default=-float('inf')) for _ in xrange(n_timepoints)]

        trace_start = [matrix(block_length + 1, block_length + 1, default=-float('inf')) for _ in xrange(n_timepoints)]
        trace_end = [matrix(block_length + 1, block_length + 1, default=-float('inf')) for _ in xrange(n_timepoints)]

        DP[0] = matcopy(emission_cache[0])
        total_log_likelihood = float('-inf')
        for t in xrange(1, n_timepoints):

            for cur_start in xrange(block_length + 1):

                for cur_end in xrange(cur_start, block_length + 1):

                    max_log_prob = -float('inf')
                    total_log_prob = -float('inf')

                    best_prev_start = None
                    best_prev_end = None

                    for prev_start in xrange(0, cur_end + 1):
                        for prev_end in xrange(max(prev_start, cur_start), block_length + 1):

                            log_prob = DP[t - 1][prev_start][prev_end] + \
                                       self.boundary_movement_model(prev_start - cur_start, t - 1, return_log=True) + \
                                       self.boundary_movement_model(cur_end - prev_end, t - 1, return_log=True)

                            total_log_prob = add_log_probs(total_log_prob, log_prob)

                            if log_prob > max_log_prob:
                                best_prev_start = prev_start
                                best_prev_end = prev_end
                                max_log_prob = log_prob

                    max_log_prob += emission_cache[t][cur_start][cur_end]

                    total_log_likelihood = add_log_probs(total_log_likelihood,
                                                         total_log_prob + emission_cache[t][cur_start][cur_end])

                    DP[t][cur_start][cur_end] = max_log_prob

                    trace_start[t][cur_start][cur_end] = best_prev_start
                    trace_end[t][cur_start][cur_end] = best_prev_end

        positions = {LEFT_BOUNDARY: [None] * n_timepoints,
                     RIGHT_BOUNDARY: [None] * n_timepoints}

        best_start, best_end = max([(s, e) for s in xrange(block_length + 1) for e in xrange(s, block_length + 1)],
                                   key=lambda (s, e): DP[n_timepoints - 1][s][e])

        combo_likelihood = DP[n_timepoints - 1][best_start][best_end]

        block_offset = block[BLOCK_OFFSET]

        positions[LEFT_BOUNDARY][n_timepoints - 1] = block_offset + best_start * self.bin_size
        positions[RIGHT_BOUNDARY][n_timepoints - 1] = block_offset + best_end * self.bin_size

        left_trajectory = [None] * (n_timepoints - 1)
        right_trajectory = [None] * (n_timepoints - 1)

        for t in reversed(xrange(1, n_timepoints)):

            prev_best_start = trace_start[t][best_start][best_end]
            prev_best_end = trace_end[t][best_start][best_end]

            positions[LEFT_BOUNDARY][t - 1] = block_offset + prev_best_start * self.bin_size
            positions[RIGHT_BOUNDARY][t - 1] = block_offset + prev_best_end * self.bin_size

            left_trajectory[t - 1] = STEADY if prev_best_start == best_start else EXPAND if prev_best_start > best_start else CONTRACT
            right_trajectory[t - 1] = STEADY if prev_best_end == best_end else EXPAND if prev_best_end < best_end else CONTRACT

            best_start = prev_best_start
            best_end = prev_best_end

        # determine the first and the last time points

        first_timepoint = None
        last_timepoint = None

        for t in xrange(n_timepoints):
            if positions[LEFT_BOUNDARY][t] != positions[RIGHT_BOUNDARY][t]:
                if first_timepoint is None:
                    first_timepoint = t
                last_timepoint = t

        return total_log_likelihood, combo_likelihood, left_trajectory, right_trajectory, positions, first_timepoint, last_timepoint

    def split_block(self, block, side, side_start, side_end, side_offset):

        if side == LEFT_BOUNDARY:
            flip = lambda a: list(reversed(a))
        else:
            flip = lambda a: a

        split_block = {FOREGROUND_SIGNAL: [flip(block[FOREGROUND_SIGNAL][t][side_start: side_end])
                                            for t in xrange(self.n_timepoints)],

                       BLOCK_COVARIATES: [flip(block[BLOCK_COVARIATES][t][side_start: side_end])
                                            for t in xrange(self.n_timepoints)],

                       CHROMOSOME: block[CHROMOSOME],

                       BLOCK_LENGTH: side_end - side_start,

                       BLOCK_OFFSET: block[BLOCK_OFFSET] + side_offset * self.bin_size,

                       BLOCK_ID: block[BLOCK_ID] + '.' + side
                       }

        return split_block

    def decode_position_path_split_C(self, block, split_point, is_peak_timepoint, path_method):

        total_log_likelihood = 0
        combo_likelihood = 0

        positions = {LEFT_BOUNDARY: [],
                     RIGHT_BOUNDARY: []}

        trajectories = {LEFT_BOUNDARY: [],
                        RIGHT_BOUNDARY: []}

        first_timepoint = None
        last_timepoint = None

        C_is_peak_timepoint = (ctypes.c_int * self.n_timepoints)(*is_peak_timepoint)

        for side, side_start, side_end in [(LEFT_BOUNDARY, 0, split_point),
                                           (RIGHT_BOUNDARY, split_point, block[BLOCK_LENGTH])]:

            split_block = self.split_block(block, side, side_start, side_end, split_point)

            if split_block[BLOCK_LENGTH] > 0:

                C_total_log_likelihood = (C_DOUBLE * 1)(0.)

                C_combo_likelihood = (C_DOUBLE * 1)(0.)
                C_trajectories = (ctypes.c_int * (self.n_timepoints - 1))()
                C_positions = (ctypes.c_int * (self.n_timepoints))()
                C_time_frame = (ctypes.c_int * 2)(0, 0)

                (C_block_fgr,
                 C_block_covariates,
                 C_block_length,

                 C_foreground_delta,
                 C_foreground_betas,

                 C_background_delta,
                 C_background_betas,

                 C_dynamics_params,
                 C_n_dynamics,
                 C_priors) = self.get_model_parameters_as_C_types(split_block)

                status = path_method( C_block_fgr,
                                      C_block_covariates,
                                      -1 if side == LEFT_BOUNDARY else 1,

                                      C_block_length,

                                      split_block[BLOCK_OFFSET],

                                      self.n_covariates,
                                      self.n_timepoints,

                                      C_foreground_delta,
                                      C_foreground_betas,

                                      C_background_delta,
                                      C_background_betas,

                                      C_dynamics_params,
                                      C_n_dynamics,

                                      C_priors,

                                      self.bin_size,

                                      C_total_log_likelihood,
                                      C_combo_likelihood,
                                      C_trajectories,
                                      C_positions,
                                      C_time_frame,
                                      C_is_peak_timepoint)

                if status == 1:
                    raise UnderflowException

                positions[side] = [C_positions[t] for t in xrange(self.n_timepoints)]

                trajectories[side] = [self.dynamics[C_trajectories[t]] for t in xrange(self.n_timepoints - 1)]

                if C_time_frame[0] != -1:
                    if first_timepoint is None:
                        first_timepoint = C_time_frame[0]
                    else:
                        first_timepoint = min(first_timepoint, C_time_frame[0])

                if C_time_frame[1] != -1:
                    if last_timepoint is None:
                        last_timepoint = C_time_frame[1]
                    else:
                        last_timepoint = max(last_timepoint, C_time_frame[1])

                total_log_likelihood += C_total_log_likelihood[0]
                combo_likelihood += C_combo_likelihood[0]
            else:
                side_log_likelihood = sum(math.log(steady_prior) for steady_prior in self.dynamic_priors[0])
                combo_likelihood += side_log_likelihood
                total_log_likelihood += side_log_likelihood

                positions[side] = [split_block[BLOCK_OFFSET] for _ in xrange(self.n_timepoints)]

                trajectories[side] = [STEADY for t in xrange(self.n_timepoints - 1)]

        return (combo_likelihood,
                total_log_likelihood,
                trajectories[LEFT_BOUNDARY],
                trajectories[RIGHT_BOUNDARY],
                positions,
                first_timepoint,
                last_timepoint)

    def compute_Viterbi_path_split_on_positions_C(self, block, is_peak_timepoint):

        # compute the Viterbi paths over all possible splits from position 1 to position block_length - 1
        # and return the paths from the split with the highest likelihood

        split_paths = [self.decode_position_path_split_C(block,
                                                         split_point,
                                                         is_peak_timepoint=is_peak_timepoint,
                                                         path_method=clib.compute_Viterbi_path_split)
                       for split_point in block[SPLIT_POINT]]
        return max(split_paths)

    def compute_posterior_path_split_on_positions_C(self, block, is_peak_timepoint):

        # compute the Viterbi paths over all possible splits from position 1 to position block_length - 1
        # and return the paths from the split with the highest likelihood

        split_paths = [self.decode_position_path_split_C(block,
                                                         split_point,
                                                         is_peak_timepoint=is_peak_timepoint,
                                                         path_method=clib.compute_posterior_path_split)
                       for split_point in block[SPLIT_POINT]]
        return max(split_paths)

    def compute_Viterbi_path_C(self, block, is_peak_timepoint):

        C_is_peak_timepoint = (ctypes.c_int * self.n_timepoints)(*is_peak_timepoint)

        C_total_log_likelihood = (C_DOUBLE * 1)(0.)

        C_combo_likelihood = (C_DOUBLE * 1)(0.)
        C_trajectories = (ctypes.c_int * (2 * (self.n_timepoints - 1)))()
        C_positions = (ctypes.c_int * (2 * self.n_timepoints))()
        C_time_frame = (ctypes.c_int * 2)(0, 0)

        (C_block_fgr,
         C_block_covariates,
         C_block_length,

         C_foreground_delta,
         C_foreground_betas,

         C_background_delta,
         C_background_betas,

         C_dynamics_params,
         C_n_dynamics,
         C_priors) = self.get_model_parameters_as_C_types(block)

        status = clib.compute_Viterbi_path(C_block_fgr,
                                           C_block_covariates,
                                           C_block_length,
                                           block[BLOCK_OFFSET],

                                           self.n_covariates,
                                           self.n_timepoints,

                                           C_foreground_delta,
                                           C_foreground_betas,

                                           C_background_delta,
                                           C_background_betas,

                                           C_dynamics_params,
                                           C_n_dynamics,

                                           C_priors,

                                           self.bin_size,

                                           C_total_log_likelihood,
                                           C_combo_likelihood,
                                           C_trajectories,
                                           C_positions,
                                           C_time_frame,
                                           C_is_peak_timepoint)

        if status == 1:
            raise UnderflowException

        positions = {LEFT_BOUNDARY: [C_positions[t] for t in xrange(self.n_timepoints)],
                     RIGHT_BOUNDARY: [C_positions[self.n_timepoints + t] for t in xrange(self.n_timepoints)]}


        left_trajectory = [self.dynamics[C_trajectories[t]] for t in xrange(self.n_timepoints - 1)]
        right_trajectory = [self.dynamics[C_trajectories[self.n_timepoints - 1 + t]] for t in xrange(self.n_timepoints - 1)]

        first_timepoint = None if C_time_frame[0] == -1 else C_time_frame[0]
        last_timepoint = None if C_time_frame[1] == -1 else C_time_frame[1]


        # We need to orient ties at the first_timepoint and the last_timepoint (i.e. creation and removal of peaks)
        # in the direction of the overal dynamic if it does not contain contradictions

        def get_direction_dynamic(traj):
            # in case of consistent dynamic trajectories this method returns the dynamic for each side
            # it returns None for mixed E/C trajectories
            dyns = sorted(set([d for d in traj]))
            if len(dyns) <= 2:
                # this will return expand or contract if there are two dynamics because dyns is sorted!
                return dyns[0]
            else:
                return None

        if first_timepoint != last_timepoint:
            # for consistent non-singletons set the orientation of the "creation" and the "removal" dynamic accordingly
            left_dyn = get_direction_dynamic(left_trajectory[first_timepoint:last_timepoint])
            right_dyn = get_direction_dynamic(right_trajectory[first_timepoint:last_timepoint])

            if first_timepoint != 0:
                if left_dyn in [EXPAND, CONTRACT] and right_dyn == STEADY:
                    if left_trajectory[first_timepoint - 1] == STEADY and right_trajectory[first_timepoint - 1] == EXPAND:
                        left_trajectory[first_timepoint - 1] = EXPAND
                        right_trajectory[first_timepoint - 1] = STEADY

                elif left_dyn == STEADY and right_dyn in [EXPAND, CONTRACT]:
                    if left_trajectory[first_timepoint - 1] == EXPAND and right_trajectory[first_timepoint - 1] == STEADY:
                        left_trajectory[first_timepoint - 1] = STEADY
                        right_trajectory[first_timepoint - 1] = EXPAND

            if last_timepoint != self.n_timepoints - 1:
                if left_dyn in [EXPAND, CONTRACT] and right_dyn == STEADY:
                    if left_trajectory[last_timepoint] == STEADY and right_trajectory[last_timepoint] == CONTRACT:
                        left_trajectory[last_timepoint] = CONTRACT
                        right_trajectory[last_timepoint] = STEADY

                elif left_dyn == STEADY and right_dyn in [EXPAND, CONTRACT]:
                    if left_trajectory[last_timepoint] == CONTRACT and right_trajectory[last_timepoint] == STEADY:
                        left_trajectory[last_timepoint] = STEADY
                        right_trajectory[last_timepoint] = CONTRACT

        elif first_timepoint > 0 and last_timepoint < self.n_timepoints - 1:
            # for intermediate singletons make "creation" and "removal" dynamics consistent

            if (left_trajectory[first_timepoint - 1] == STEADY and right_trajectory[last_timepoint] == STEADY or
                    right_trajectory[first_timepoint - 1] == STEADY and left_trajectory[last_timepoint] == STEADY):

                left_trajectory[first_timepoint - 1], right_trajectory[first_timepoint - 1] = \
                    right_trajectory[first_timepoint - 1], left_trajectory[first_timepoint - 1]

        return C_total_log_likelihood[0], C_combo_likelihood[0], left_trajectory, right_trajectory, positions, first_timepoint, last_timepoint

    def compute_posterior_paths_C(self, block, is_peak_timepoint):

        C_is_peak_timepoint = (ctypes.c_int * self.n_timepoints)(*is_peak_timepoint)

        C_total_log_likelihood = (C_DOUBLE * 1)(0.)

        C_combo_likelihood = (C_DOUBLE * 1)(0.)
        C_trajectories = (ctypes.c_int * (2 * (self.n_timepoints - 1)))()
        C_positions = (ctypes.c_int * (2 * self.n_timepoints))()
        C_time_frame = (ctypes.c_int * 2)(0, 0)

        (C_block_fgr,
         C_block_covariates,
         C_block_length,

         C_foreground_delta,
         C_foreground_betas,

         C_background_delta,
         C_background_betas,

         C_dynamics_params,
         C_n_dynamics,
         C_priors) = self.get_model_parameters_as_C_types(block)

        status = clib.compute_posterior_paths(C_block_fgr,
                                           C_block_covariates,
                                           C_block_length,
                                           block[BLOCK_OFFSET],

                                           self.n_covariates,
                                           self.n_timepoints,

                                           C_foreground_delta,
                                           C_foreground_betas,

                                           C_background_delta,
                                           C_background_betas,

                                           C_dynamics_params,
                                           C_n_dynamics,

                                           C_priors,

                                           self.bin_size,

                                           C_total_log_likelihood,
                                           C_combo_likelihood,
                                           C_trajectories,
                                           C_positions,
                                           C_time_frame,
                                           C_is_peak_timepoint)

        if status == 1:
            raise UnderflowException

        positions = {LEFT_BOUNDARY: [C_positions[t] for t in xrange(self.n_timepoints)],
                     RIGHT_BOUNDARY: [C_positions[self.n_timepoints + t] for t in xrange(self.n_timepoints)]}


        left_trajectory = [self.dynamics[C_trajectories[t]] for t in xrange(self.n_timepoints - 1)]
        right_trajectory = [self.dynamics[C_trajectories[self.n_timepoints - 1 + t]] for t in xrange(self.n_timepoints - 1)]

        first_timepoint = None if C_time_frame[0] == -1 else C_time_frame[0]
        last_timepoint = None if C_time_frame[1] == -1 else C_time_frame[1]

        return C_total_log_likelihood[0], C_combo_likelihood[0], left_trajectory, right_trajectory, positions, first_timepoint, last_timepoint

    def compute_no_peak_probability(self, block):

        n_timepoints = self.n_timepoints

        block_no_peak_probs = [None] * n_timepoints

        if block[BLOCK_LENGTH] > self.max_region_length:
            split_no_peak_probs = [[] for _ in xrange(n_timepoints)]
            split_likelihoods = []

            for split_point in block[SPLIT_POINT]:

                # print 'SPLIT:', split_point

                total_split_log_likelihood = 0
                split_no_peak_likelihoods = [0] * n_timepoints

                for side, side_start, side_end in [(LEFT_BOUNDARY, 0, split_point),
                                                   (RIGHT_BOUNDARY, split_point, block[BLOCK_LENGTH])]:

                    split_block = self.split_block(block, side, side_start, side_end, split_point)

                    if split_block[BLOCK_LENGTH] > 0:

                        C_total_split_log_likelihood = (C_DOUBLE * 1)(0.)

                        C_side_no_peak_likelihoods = (C_DOUBLE * n_timepoints)(0.)

                        (C_block_fgr,
                         C_block_covariates,
                         C_block_length,

                         C_foreground_delta,
                         C_foreground_betas,

                         C_background_delta,
                         C_background_betas,

                         C_dynamics_params,
                         C_n_dynamics,
                         C_priors) = self.get_model_parameters_as_C_types(split_block)

                        status = clib.get_no_peak_likelihoods_split(C_block_fgr,

                                                                    C_block_covariates,

                                                                    C_block_length,

                                                                    self.n_covariates,
                                                                    self.n_timepoints,

                                                                    C_foreground_delta,
                                                                    C_foreground_betas,

                                                                    C_background_delta,
                                                                    C_background_betas,

                                                                    C_dynamics_params,
                                                                    C_n_dynamics,

                                                                    C_priors,

                                                                    C_total_split_log_likelihood,
                                                                    C_side_no_peak_likelihoods)

                        if status == 1:
                            raise UnderflowException

                        total_split_log_likelihood += C_total_split_log_likelihood[0]

                        for t in xrange(n_timepoints):
                            split_no_peak_likelihoods[t] += C_side_no_peak_likelihoods[t]

                    else:
                        side_log_likelihood = sum(math.log(steady_prior) for steady_prior in self.dynamic_priors[0])
                        for t in xrange(n_timepoints):
                            split_no_peak_likelihoods[t] += side_log_likelihood

                        total_split_log_likelihood += side_log_likelihood

                for t in xrange(n_timepoints):
                    split_no_peak_probs[t].append(math.exp(split_no_peak_likelihoods[t] - total_split_log_likelihood))

                split_likelihoods.append(total_split_log_likelihood)

            split_posteriors = convert_and_normalize_log_posteriors(split_likelihoods)

            for t in xrange(n_timepoints):
                block_no_peak_probs[t] = sum(no_peak_prob * split_posterior
                                             for no_peak_prob, split_posterior
                                             in izip(split_no_peak_probs[t], split_posteriors))

        else:

            C_total_log_likelihood = (C_DOUBLE * 1)(0.)

            C_no_peak_likelihoods = (C_DOUBLE * n_timepoints)(0.)

            (C_block_fgr,
             C_block_covariates,
             C_block_length,

             C_foreground_delta,
             C_foreground_betas,

             C_background_delta,
             C_background_betas,

             C_dynamics_params,
             C_n_dynamics,
             C_priors) = self.get_model_parameters_as_C_types(block)

            status = clib.get_no_peak_likelihoods(C_block_fgr,

                                                  C_block_covariates,

                                                  C_block_length,

                                                  self.n_covariates,
                                                  self.n_timepoints,

                                                  C_foreground_delta,
                                                  C_foreground_betas,

                                                  C_background_delta,
                                                  C_background_betas,

                                                  C_dynamics_params,
                                                  C_n_dynamics,

                                                  C_priors,

                                                  C_total_log_likelihood,
                                                  C_no_peak_likelihoods)

            if status == 1:
                raise UnderflowException

            total_log_likelihood = C_total_log_likelihood[0]

            for t in xrange(n_timepoints):
                block_no_peak_probs[t] = math.exp(C_no_peak_likelihoods[t] - total_log_likelihood)

        return block_no_peak_probs


def get_read_counts(reads_fname, bins, shift):
    read_counts = {}

    echo('Counting reads in ' + reads_fname)
    total_aligned_reads_per_chromosome = {}

    with open_file(reads_fname) as aln_f:
        for l in aln_f:
            chrom = l.split('\t')[0]
            if chrom not in total_aligned_reads_per_chromosome:
                total_aligned_reads_per_chromosome[chrom] = 0

            total_aligned_reads_per_chromosome[chrom] += 1

    total_aligned_reads = sum(total_aligned_reads_per_chromosome.values())
    echo('Total aligned reads: ' + str(total_aligned_reads))
    # print pprint.pformat(total_aligned_reads_per_chromosome)

    reads = dict((chrom, [0] * total_aligned_reads_per_chromosome[chrom]) for chrom in total_aligned_reads_per_chromosome)
    cur_read_idx = dict((chrom, 0) for chrom in total_aligned_reads_per_chromosome)

    with open_file(reads_fname) as aln_f:

        for l in aln_f:
            buf = l.strip().split('\t')

            chrom = buf[0]
            start = int(buf[1])
            end = int(buf[2])

            strand = buf[-1]

            if strand == '+':
                read_start = start + shift
            else:
                read_start = end - shift

            reads[chrom][cur_read_idx[chrom]] = read_start
            cur_read_idx[chrom] += 1

    # sort the reads by starting position
    for chrom in sorted(bins):
        chrom_reads = sorted(reads[chrom]) if chrom in reads else []
        c_read_idx = 0
        for start, end, source in bins[chrom]:
            # print chrom, start, end, source

            n_reads = 0
            while c_read_idx < len(chrom_reads) and chrom_reads[c_read_idx] < start:
                # print chrom_reads[c_read_idx]
                c_read_idx += 1

            while c_read_idx < len(chrom_reads) and chrom_reads[c_read_idx] < end:

                # print c_read_idx, chrom_reads[c_read_idx], n_reads + 1

                c_read_idx += 1
                n_reads += 1

            if source not in read_counts:
                read_counts[source] = {CHROMOSOME: chrom,
                                       BLOCK_OFFSET: start,
                                       SIGNAL: []}

            read_counts[source][BLOCK_OFFSET] = min(read_counts[source][BLOCK_OFFSET], start)
            read_counts[source][SIGNAL].append(n_reads)

    return read_counts, total_aligned_reads


def dynamic_color(left_dynamic, right_dynamic):

    # return black for UNKNOWN
    if left_dynamic is None and right_dynamic is None:
        return '0,0,0'

    dynamics = set([left_dynamic, right_dynamic])

    if EXPAND in dynamics and (STEADY in dynamics or len(dynamics) == 1):
        return '255,0,0'
    elif CONTRACT in dynamics and (STEADY in dynamics or len(dynamics) == 1):
        return '0,0,255'

    elif STEADY in dynamics and len(dynamics) == 1:
        return '192,192,192'
    else:
        return '0,0,0'


def worker(args):
    func = None

    try:
        func = args[0]
        return func(*args[1:])

    except Exception, e:
        print 'Caught exception in output worker thread (pid: %d):' % os.getpid()
        print func

        echo(e)
        if hasattr(open_log, 'logfile'):
            traceback.print_exc(file=open_log.logfile)
        traceback.print_exc()

        print
        raise e


def compute_no_peak_probability(theModel, batch, batch_no):
    theModel.init_caches(batch)
    n_timepoints = theModel.n_timepoints

    batch_no_peak_probabilities = [[] for _ in xrange(n_timepoints)]
    print >>sys.stderr, ".",

    for block in batch:
            block_no_peak_probs = theModel.compute_no_peak_probability(block)
            for t in xrange(n_timepoints):
                batch_no_peak_probabilities[t].append(block_no_peak_probs[t])

    theModel.free_caches()

    return batch_no_peak_probabilities


def EM_batch(model, batch, batch_no):
    underflows = []

    batch_param_info = model.new_param_info()

    model.init_caches(batch)

    total_likelihood = 0

    for b_idx, block in enumerate(batch):
        if b_idx % 100 == 1:
            print >>sys.stderr, '.',

        block_likelihood = None

        try:
            # model.clear_param_info(block_param_info)
            block_param_info = model.new_param_info(block)
            if block[BLOCK_LENGTH] > model.max_region_length:
                EM_step_func = model.EM_step_split_C
            else:
                EM_step_func = model.EM_step_C

            block_likelihood = EM_step_func(block, block_param_info)

            model.add_param_info(block_param_info, batch_param_info)

            total_likelihood += block_likelihood

        except UnderflowException:
            print 'underflow:', batch_no, b_idx, block
            underflows.append(block[BLOCK_ID])

        if any(math.isnan(v) or math.isinf(v) for d in xrange(model.n_dynamics) for v in block_param_info[TOTAL_POSTERIORS_PER_DYNAMIC][d]):
            print b_idx, block[BLOCK_ID], block
            exit(1)

        if block_likelihood is not None and (math.isinf(block_likelihood) or math.isnan(block_likelihood)):
            print 'log likelihood is nan:', block_likelihood, b_idx, block[BLOCK_ID], block
            exit(1)

    model.free_caches()
    return batch_param_info, underflows, total_likelihood


def extract_blocks(blocks_fname,
                   aligned_fnames,
                   control_fnames,

                   shift,

                   bin_size):

    block_bins = {}
    echo('Reading the block boundaries from:', blocks_fname)

    EXTEND = 1000

    with open_file(blocks_fname) as in_f:
        _blocks = []
        for l in in_f:
            buf = l.strip().split('\t')

            chrom, start, end, reg_id = buf[:4]
            start = int(start)
            end = int(end)

            _blocks.append([chrom, start, end, reg_id])

        _blocks = sorted(_blocks)

        for i in xrange(len(_blocks)):
            chrom, start, end, reg_id = _blocks[i]

            if i == 0:
                start = max(0, start - EXTEND)
            elif i == len(_blocks) - 1:
                end += EXTEND
            else:
                prev_chrom, prev_start, prev_end, prev_reg_id = _blocks[i - 1]
                if prev_chrom != chrom:
                    prev_end += EXTEND
                    start = max(0, start - EXTEND)
                elif prev_end + EXTEND < start - EXTEND:
                    prev_end += EXTEND
                    start = max(0, start - EXTEND)
                else:
                    mid_point = (prev_end + start) / 2
                    prev_end = mid_point
                    start = mid_point + 1
                _blocks[i - 1] = [prev_chrom, prev_start, prev_end, prev_reg_id]

            _blocks[i] = [chrom, start, end, reg_id]

        for chrom, start, end, reg_id in _blocks:

            if chrom not in block_bins:
                block_bins[chrom] = []

            for bin_start in xrange((start / bin_size) * bin_size,
                                    (end / bin_size) * bin_size + 1,
                                    bin_size):

                block_bins[chrom].append([bin_start,
                                          bin_start + bin_size,
                                          reg_id])

    all_bins = dict((chrom, sorted(block_bins[chrom])) for chrom in block_bins)

    all_blocks = {}
    total_fgr_reads = []
    total_bgr_reads = []
    n_timepoints = len(aligned_fnames)

    n_covariates = 2

    for aln_fname, ctrl_fname in zip(aligned_fnames, control_fnames):
        fgr_read_count, n_fgr_reads = get_read_counts(aln_fname, all_bins, shift)

        if ctrl_fname is not None:
            bgr_read_count, n_bgr_reads = get_read_counts(ctrl_fname, all_bins, shift)
        else:
            raise Exception("Control reads are required")

        total_fgr_reads.append(n_fgr_reads)
        total_bgr_reads.append(n_bgr_reads)

        for source in fgr_read_count:
            block_length = len(fgr_read_count[source][SIGNAL])

            if source not in all_blocks:
                all_blocks[source] = {FOREGROUND_SIGNAL: [],
                                      BACKGROUND_SIGNAL: [],
                                      BLOCK_COVARIATES: cube(n_timepoints, block_length, n_covariates, default=1.),
                                      CHROMOSOME: fgr_read_count[source][CHROMOSOME],
                                      BLOCK_OFFSET: fgr_read_count[source][BLOCK_OFFSET],
                                      BLOCK_LENGTH: block_length,
                                      IS_SUBPEAK: '-sub_' in source,
                                      BLOCK_ID: source}

            t_idx = len(all_blocks[source][FOREGROUND_SIGNAL])
            all_blocks[source][FOREGROUND_SIGNAL].append(fgr_read_count[source][SIGNAL])
            all_blocks[source][BACKGROUND_SIGNAL].append(bgr_read_count[source][SIGNAL])

            BGR_PSEUDO_COUNT = 1
            expected_reads_per_bin = BGR_PSEUDO_COUNT + (n_fgr_reads / float(n_bgr_reads)) * \
                                                        float(sum(bgr_read_count[source][SIGNAL])) / block_length

            for p in xrange(block_length):
                all_blocks[source][BLOCK_COVARIATES][t_idx][p][0] = math.log(expected_reads_per_bin)

    return all_blocks


def store_classification(theModel, full_output_fname, out_fnames, blocks, batch_no, output_empty_blocks):

    out_files = [open_file(fname + '.' + str(batch_no) + '.gz', 'w') for fname in out_fnames]

    full_output_f = open_file(full_output_fname + '.' + str(batch_no) + '.gz', 'w')

    theModel.init_caches(blocks)

    for reg_no, block in enumerate(blocks):

        block_id = block[BLOCK_ID]
        chrom = block[CHROMOSOME]

        if reg_no % 500 == 0:
            print >> sys.stderr, '.',

        if block[BLOCK_LENGTH] > theModel.max_region_length:

            path_method = theModel.compute_Viterbi_path_split_on_positions_C
            posteriors_method = theModel.compute_posteriors_split_on_positions_C
        else:

            path_method = theModel.compute_Viterbi_path_C
            posteriors_method = theModel.compute_posteriors_C

        no_peak_posteriors = theModel.compute_no_peak_probability(block)

        dynamics_posteriors = posteriors_method(block)
        is_peak_timepoint = [int(no_peak_posteriors[t] <= theModel.fdr_threshold_for_decoding[t])
                              for t in xrange(theModel.n_timepoints)]

        (combo_likelihood,
         total_log_likelihood,
         left_trajectory,
         right_trajectory,
         positions,
         first_timepoint,
         last_timepoint) = path_method(block, is_peak_timepoint)

        if first_timepoint is not None or output_empty_blocks:
            if first_timepoint is None:
                first_timepoint = last_timepoint = -1

            for t_idx in xrange(theModel.n_timepoints - 1):
                if t_idx < first_timepoint - 1 or t_idx > last_timepoint:
                    left_trajectory[t_idx] = 'x'
                    right_trajectory[t_idx] = 'x'

            left_trajectory_label = '-'.join(left_trajectory)
            right_trajectory_label = '-'.join(right_trajectory)

            for t_idx in xrange(first_timepoint, last_timepoint + 1):
                f = out_files[t_idx]
                if t_idx == first_timepoint:
                    if first_timepoint != last_timepoint:
                        # color first peak in dark grey
                        peak_color = '90,90,90'
                    else:
                        # return light orange for singletons
                        peak_color = '255,204,153'
                else:
                    peak_color = dynamic_color(left_trajectory[t_idx - 1],
                                               right_trajectory[t_idx - 1])

                # skip empty peaks in the middle of the time course
                if positions[LEFT_BOUNDARY][t_idx] != positions[RIGHT_BOUNDARY][t_idx]:
                    f.write('\t'.join(map(str, [chrom,

                                                positions[LEFT_BOUNDARY][t_idx],
                                                positions[RIGHT_BOUNDARY][t_idx],

                                                block_id + '#' + left_trajectory_label + '/' + right_trajectory_label,

                                                '1000',
                                                '.',

                                                positions[LEFT_BOUNDARY][t_idx],
                                                positions[RIGHT_BOUNDARY][t_idx],

                                                peak_color])) + '\n')

            full_output_f.write('\t'.join(map(str, [chrom,
                                                    min(positions[LEFT_BOUNDARY]),
                                                    max(positions[RIGHT_BOUNDARY]),
                                                    block_id,
                                                    left_trajectory_label + '/' + right_trajectory_label,
                                                    combo_likelihood,
                                                    total_log_likelihood,
                                                    first_timepoint,
                                                    last_timepoint,
                                                    'posteriors:' + str(dynamics_posteriors),
                                                    'predicted_starts:' + ','.join(map(str, positions[LEFT_BOUNDARY])),
                                                    'predicted_ends:' + ','.join(map(str, positions[RIGHT_BOUNDARY]))
                                                     ])) + '\n')

    theModel.free_caches()
    map(lambda f: f.close(), out_files)
    full_output_f.close()

    return batch_no


def call_boundary_dynamics(blocks,

                           bin_size,

                           model_fname,

                           n_threads,
                           skip_training,

                           n_training_examples,
                           max_region_length,

                           aligned_fnames,
                           out_prefix,
                           fdr_for_decoding=0.05,
                           output_empty_blocks=False):

    _blk = blocks.values()[0]
    n_timepoints = len(_blk[FOREGROUND_SIGNAL])
    n_covariates = len(_blk[BLOCK_COVARIATES][0][0])

    echo('Total blocks:', len(blocks))
    echo('Max block length:', max(b[BLOCK_LENGTH] for b in blocks.itervalues()))
    echo('Blocks longer than %d bins:' % max_region_length,
         len([1 for b in blocks.itervalues() if b[BLOCK_LENGTH] > max_region_length]))

    echo('Sub-peak blocks :' , len([b for b in blocks if blocks[b][IS_SUBPEAK]]))

    echo('n_timepoints:', n_timepoints)
    echo('n_covariates:', n_covariates)

    block_ids_for_training = [block_id for block_id in sorted(blocks) if not blocks[block_id][IS_SUBPEAK]]

    if n_training_examples != -1:
        block_ids_for_training = random.sample(block_ids_for_training,
                                               min(n_training_examples, len(block_ids_for_training)))

    echo('n_training_examples:', len(block_ids_for_training))

    theModel = ClusterModel(n_timepoints,
                            out_prefix,
                            n_threads,
                            max_region_length,
                            bin_size=bin_size,
                            n_covariates=n_covariates)

    if model_fname is not None:
        theModel.init_from_file(model_fname)

    theModel.print_model()

    if not skip_training:
        theModel.EM(dict((block_id, blocks[block_id]) for block_id in block_ids_for_training))
        # theModel.EM_bruteforce_debug(dict((block_id, blocks[block_id]) for block_id in block_ids_for_training))

    # theModel.EM_bruteforce_debug(dict((block_id, blocks[block_id]) for block_id in block_ids_for_training),
    #             total_fgr_reads,
    #             total_bgr_reads)

    # echo("WARNING!!! PRESET FDR FOR DECODING!")
    # theModel.reset_model()
    # theModel.fdr_threshold_for_decoding = [0.011160035126507857, 0.008515206674514574, 0.012605835237376196,
    #                                        0.015018265584743291]

    if not hasattr(theModel, 'fdr_threshold_for_decoding') or any(fdrt is None for fdrt in theModel.fdr_threshold_for_decoding):
        theModel.compute_FDR_threshold(blocks, fdr=fdr_for_decoding)
        theModel.save_model()

    out_fnames = [out_prefix + '.' + os.path.split(fname)[1].replace('.bed', '').replace('.gz', '') +
                      '.ChromTime_timepoint_predictions.bed.gz' for fname in aligned_fnames]

    out_files = [open(fname, 'w') for fname in out_fnames]
    full_output_fname = out_prefix + '.ChromTime_full_output.bed.gz'

    full_output_f = open(full_output_fname, 'w')

    echo('Storing output')

    if n_threads > 1:
        pool = Pool(processes=n_threads)
        _map = pool.map
    else:
        _map = map

    CHUNK_SIZE = 500

    def chunks(array, size):
        cur_idx = 0
        while cur_idx < len(array):
            end_idx = cur_idx + size
            while end_idx < len(array) and array[end_idx][IS_SUBPEAK]:
                end_idx += 1

            yield array[cur_idx: end_idx]

            cur_idx = end_idx

    for batch_no in \
            _map(worker, [(store_classification, theModel, full_output_fname, out_fnames, batch, batch_no, output_empty_blocks)
                          for batch_no, batch in enumerate(chunks(sorted(blocks.values(), key=lambda b: b[BLOCK_ID]),
                                                                  CHUNK_SIZE))]):

        for fname in [full_output_fname] + out_fnames:
            append_and_unlink(fname + '.' + str(batch_no) + '.gz',
                              fname)

    print >> sys.stderr, ''
    map(lambda f: f.close(), out_files)
    full_output_f.close()
    echo('Output stored in:', out_prefix)

    if n_threads > 1:
        pool.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fix peak boundaries')

    parser.add_argument("-i", "--block-boundaries", dest="blocks_fname",
                      help="BED file with block boundaries", metavar="FILE")

    parser.add_argument("-o", "--output-dir", dest="out_dir",
                      help="Output directory", metavar="DIRECTORY")

    parser.add_argument("-b", "--bin-size", type=int, dest="bin_size", default=200,
                      help="Bin size in base pairs [%(default)s]", metavar="INT")

    parser.add_argument("-t", "--threads", type=int, dest="n_threads", default=1,
                      help="Number of threads to use [%(default)s]",
                      metavar="INT")

    parser.add_argument("-n", "--n-training-examples", type=int, dest="n_training_examples", default=20000,
                      help="Number of training examples to use. Default: all",
                      metavar="INT")

    parser.add_argument("--max-region-length", type=int, dest="max_region_length", default=5000,
                      help="Maximum outer box length",
                      metavar="INT")

    parser.add_argument("-m", "--model-file", dest="model_fname",
                      help="Pickled model file to load a learned the model from", metavar="FILE")

    parser.add_argument("--skip-training", action="store_true", dest="skip_training", default=False,
                      help="Skip EM training [%(default)s]")

    parser.add_argument('-s',
                        '--shift',
                        type=int,
                        dest='shift',
                        help='Number of bases to shift each read (Default: %(default)s)',
                        default=100)

    parser.add_argument('-a',
                        '--aligned-reads',
                        dest='aligned_fnames',
                        nargs='+',
                        help='Bed files with aligned reads for each time point in the correct order')

    parser.add_argument('-c',
                        '--control-reads',
                        dest='control_fnames',
                        nargs='+',
                        help='Bed files with aligned reads for control (input) for each time point in the correct order')

    args = parser.parse_args()

    # if no options were given by the user, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    out_prefix = os.path.join(args.out_dir, os.path.split(args.blocks_fname)[1].replace('.bed', '').replace('.gz', ''))
    open_log(out_prefix + '.log')

    echo('BIN_SIZE:', args.bin_size)
    echo('Command line:', ' '.join(sys.argv), level=ECHO_TO_LOGFILE)

    # You should always set the random seed in the beginning of your software
    # in order to obtain reproducible results!
    # Here we set the random seed to 42.
    # This seed tends to produce the best results, because
    # 42 is the answer to the Ultimate Question of Life, the Universe and Everything.
    # [Hitchhiker's Guide to the Galaxy, Douglas Adams]

    random.seed(42)

    echo('Extracting blocks')
    blocks = extract_blocks(blocks_fname=args.blocks_fname,
                            aligned_fnames=args.aligned_fnames,
                            control_fnames=args.control_fnames,
                            shift=args.shift,
                            bin_size=args.bin_size)

    echo('Calling boundary dynamics')
    call_boundary_dynamics(blocks,
                           bin_size=args.bin_size,

                           model_fname=args.model_fname,
                           n_threads=args.n_threads,
                           skip_training=args.skip_training,

                           n_training_examples=args.n_training_examples,
                           max_region_length=args.max_region_length / args.bin_size,

                           aligned_fnames=args.aligned_fnames,
                           out_prefix=out_prefix)
