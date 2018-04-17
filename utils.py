"""
    Here is the place to put utility methods that are shared by the modules.
"""
import gzip
from itertools import izip
import json
import os
import re
import io

import sys
import string
import datetime
import math
import ctypes

ROOT_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(ROOT_DIR, 'DATA')

# a constant that denotes all chromosome ids

# C_DOUBLE = ctypes.c_longdouble
C_DOUBLE = ctypes.c_double


def get_genome_dir(genome):
    return os.path.join(DATA_DIR, genome)


def get_target_fname(genome, target, is_binary):
    return os.path.join(os.path.join(get_genome_dir(genome),
                                     'BINARY_TARGETS' if is_binary else 'NUMERIC_TARGETS'),
                        target + '.bed.gz')


def overlap(s1, e1, s2, e2):
    return min(e1, e2) - max(s1, s2)


def error(*msg):
    echo('ERROR:', *msg)
    exit(1)


_rc_trans = string.maketrans('ACGT', 'TGCA')


def reverse_compl_seq(strseq):
    """ Returns the reverse complement of a DNA sequence
    """
    return strseq.translate(_rc_trans)[::-1]


def required(name, value):
    """ This method checks whether a required command line option was suplanted indeed.
        If not, prints out an error message and terminates the process.
    """
    if value is None:
        error("%s is required!" % name)


global_stime = datetime.datetime.now()


def elapsed(message = None):
    """ Measures how much time has elapsed since the last call of this method and the beginning of the execution.
        If 'message' is given, the message is printed out with the times.
    """
    print "[Last: " + str(datetime.datetime.now() - elapsed.stime) + ', Elapsed time: '+str(datetime.datetime.now() - global_stime)+ "] %s" % message if message is not None else ""
    elapsed.stime = datetime.datetime.now()
elapsed.stime = datetime.datetime.now()


def open_log(fname):
    """ Opens a log file
    """
    open_log.logfname = fname
    open_log.logfile = open(fname, 'w', 1)


def logm(message):
    """ Logs a message with a time stamp to the log file opened by open_log
    """
    print "[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message)
    open_log.logfile.write("[ %s ] %s\n" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))

ECHO_TO_SCREEN = 1
ECHO_TO_LOGFILE = 1
def echo(*message, **kwargs):
    to_print = "[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join(map(str, message)))
    if kwargs.get('level', ECHO_TO_SCREEN) == ECHO_TO_SCREEN:
        print to_print

    if hasattr(open_log, 'logfile'):
        open_log.logfile.write(to_print + '\n')


def echo_to_stdder(*message):
    to_print = "[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join(map(str, message)))
    print >>sys.stderr, to_print
    if hasattr(open_log, 'logfile'):
        open_log.logfile.write(to_print + '\n')

def close_log():
    """ Closes the log file opened by open_log
    """
    echo('Elapsed total time:', str(datetime.datetime.now() - global_stime))
    open_log.logfile.close()


PLOT_DIR = 'plots'


def plot_d(fname):
    return os.path.join(PLOT_DIR, fname)


def listdir(path, filter = None):
    return (os.path.join(path, fname) for fname in os.listdir(path) if filter is None or re.search(filter, fname))


def matrix(n, m, default=0, dtype=None):
    if dtype is None:
        return [[default for _ in xrange(m)] for _ in xrange(n)]
    elif dtype == 'c_double':

        import numpy
        return numpy.zeros((n, m))
    else:
        raise "Unknown dtype: " + dtype


def set_matrix(m, value):
    for i in xrange(len(m)):
        for j in xrange(len(m[i])):
            m[i][j] = value


def cube(n, m, k, default=0):
    return [[[default for _ in xrange(k)] for _ in xrange(m)] for _ in xrange(n)]


def matcopy(mat):
    return [list(row) for row in mat]


def mean(array):
    return float(sum(array)) / len(array) if len(array) > 0 else 0

def std(array):
    m = mean(array)
    return math.sqrt(sum((x - m) ** 2 for x in array) / float(len(array)))

def unbiased_std(array):
    m = mean(array)
    return math.sqrt(sum((x - m) ** 2 for x in array) / float(len(array) - 1))

def unbiased_stdder(array):
    std = unbiased_std(array)
    n = len(array)
    Cn = math.sqrt((n - 1) / 2.) * math.gamma((n-1)/2.) / math.gamma(n/2.)
    return Cn * std / math.sqrt(n)


def variance(array):
    return std(array) ** 2


def mean_and_std(array):
    m = mean(array)
    return m, math.sqrt(sum((x - m) ** 2 for x in array) / float(len(array)))

def mean_and_std_and_variance(array):
    m = mean(array)
    var = sum((x - m) ** 2 for x in array) / float(len(array))
    return m, math.sqrt(var), var


def eucld(x, y):
    return math.sqrt(sum((xx - yy) ** 2 for xx, yy in izip(x, y)))

rmse = eucld


def R2(responses, predictions):
    mean_response = mean(responses)
    var_response = sum((r - mean_response) ** 2 for r in responses)

    return 1 - (float(sum((r - p) ** 2 for r, p in izip(responses, predictions))) / var_response)


def pearsonr(responses, predictions):
    mean_response = mean(responses)
    var_response = sum((r - mean_response) ** 2 for r in responses)

    mean_prediction = mean(predictions)
    var_prediction = sum((r - mean_prediction) ** 2 for r in predictions)

    return sum((r - mean_response) * (p - mean_prediction)
                   for r, p in izip(responses, predictions)) / math.sqrt(var_response * var_prediction)


def get_params(ctrl_fname):

    with open(ctrl_fname) as control_f:
        control = json.load(control_f)

    for k in control:
        if k.endswith('fname'):
            control[k] = os.path.join(control['root'], control[k])

    return control


def read_chrom_lengths_in_bins(genome, bin_size, chrom_ids=None):

    if os.path.exists(genome):
        genome_fname = genome
    else:
        genome_fname = os.path.join(os.path.join(get_genome_dir(genome), 'CHROMOSOMES'), genome + '.txt')

    chrom_lengths = {}
    with open(genome_fname) as in_f:
        for line in in_f:
            chrom, l = line.strip().split()

            if chrom_ids is not None and chrom not in chrom_ids:
                continue

            chrom_lengths[chrom] = 1 + int(l) / bin_size

    return chrom_lengths


def state_key(state):
    try:
        return int(state[1:])
    except:
        pass
    try:
        return int(state.split('_')[0])
    except:
        return state


def poisson_pmf(k, Lambda):
    return math.exp(k * math.log(Lambda) - math.lgamma(k + 1.0) - Lambda)


def log2_poisson_pmf(k, Lambda):
    return (k * math.log(Lambda) - math.lgamma(k + 1.0) - Lambda) / math.log(2)


def normal_pdf(x, mu=0.0, stddev=1.0, return_log=False):
    u = float((x - mu) / stddev)

    if return_log:
        y = -u * u / 2 - math.log(math.sqrt(2 * math.pi) * stddev)
    else:

        y = math.exp(-u * u / 2) / (math.sqrt(2 * math.pi) * stddev)
    return y


def normal_approximation_to_binomial(x, mu, stddev, return_log=False):
    if stddev >= 1:
        return (log2_normal_pdf if return_log else normal_pdf)(x, mu, stddev)
    else:
        return normal_approximation_to_binomial_(x, mu, stddev, return_log=return_log)

def normal_cdf(x, mu=0.0, stddev=1.0):
    return (1 + math.erf((x - mu) / (stddev * math.sqrt(2)))) / 2.


def normal_approximation_to_binomial_(x, mu, stddev, return_log=False):
    if stddev == 0:
        prob = int(x == mu)
    else:
        prob = (math.erf((x + 0.5 - mu) / (stddev * math.sqrt(2))) - math.erf((x - 0.5 - mu) / (stddev * math.sqrt(2)))) / 2

    if return_log:
        if prob == 0:
            return float('-inf')
        else:
            return log2(prob)
    else:
        return prob


# def log2_normal_pdf(x, mu=0.0, stddev=1.0):
#     u = float((x - mu) / stddev)
#     y = -u * u / 2 - math.log(math.sqrt(2 * math.pi) * stddev)
#     return y / math.log(2)


def log_normal_pdf(x, mu=0.0, stddev=1.0):
    u = (x - mu) / float(stddev)
    y = -u * u / 2 - math.log(math.sqrt(2 * math.pi) * stddev)
    return y


log2 = lambda x: math.log(x, 2)


class UnderflowException(Exception):
    pass


def convert_and_normalize_log2_posteriors(log2_posteriors):
    # normalize all state posteriors to sum to one
    # find the maximum posterior
    max_log2_posterior = max(log2_posteriors)

    # calculate the ratios between all other posteriors and the maximum posterior
    posterior_ratios = [2 ** (log_post - max_log2_posterior) for log_post in log2_posteriors]

    total_posterior_ratios = sum(posterior_ratios)

    # in case of underflow return a uniform distribution
    if total_posterior_ratios == 0 or math.isinf(1. / total_posterior_ratios):
        raise UnderflowException
        # return [1. / len(log2_posteriors)] * len(log2_posteriors)

    # calculate the maximum rescaled posterior
    max_posterior = 1. / total_posterior_ratios

    # rescale cluster posteriors to sum to one
    rescaled_posteriors = [max_posterior * ratio for ratio in posterior_ratios]

    return rescaled_posteriors


def convert_and_normalize_log_posteriors(log_posteriors):
    # normalize all state posteriors to sum to one
    # find the maximum posterior
    max_log_posterior = max(log_posteriors)

    # calculate the ratios between all other posteriors and the maximum posterior
    posterior_ratios = [math.e ** (log_post - max_log_posterior) for log_post in log_posteriors]

    total_posterior_ratios = sum(posterior_ratios)

    # in case of underflow return a uniform distribution
    if total_posterior_ratios == 0 or math.isinf(1. / total_posterior_ratios):
        raise UnderflowException
        # return [1. / len(log2_posteriors)] * len(log2_posteriors)

    # calculate the maximum rescaled posterior
    max_posterior = 1. / total_posterior_ratios

    # rescale cluster posteriors to sum to one
    rescaled_posteriors = [max_posterior * ratio for ratio in posterior_ratios]

    return rescaled_posteriors


def convert_and_normalize_log_matrix(log2_matrix, log_base):
    # normalize all state posteriors to sum to one
    # find the maximum posterior

    n_rows = len(log2_matrix)
    n_cols = len(log2_matrix[0])

    all_values = [v for r in log2_matrix for v in r if not math.isinf(v)]
    max_log2_value = max(all_values)

    # calculate the ratios between all other posteriors and the maximum posterior
    posterior_ratios = [log_base ** (log_value - max_log2_value) for log_value in all_values]

    total_posterior_ratios = sum(posterior_ratios)

    # in case of underflow return a uniform distribution
    if total_posterior_ratios == 0 or math.isinf(1. / total_posterior_ratios):
        raise UnderflowException
        # scale = 1
        # for i in xrange(n_rows):
        #     for j in xrange(n_cols):
        #
        #         if not math.isinf(log2_matrix[i][j]):
        #             log2_matrix[i][j] = 1. / len(all_values)
        #         else:
        #             log2_matrix[i][j] = 0
    else:
        # calculate the maximum rescaled posterior
        max_posterior = 1. / total_posterior_ratios
        # scale = log_base ** (math.log(max_posterior, log_base) - max_log2_value)
        # scale = log2(max_posterior) - max_log2_value

        # rescale cluster posteriors to sum to one
        for i in xrange(n_rows):
            for j in xrange(n_cols):

                if not math.isinf(log2_matrix[i][j]):
                    log2_matrix[i][j] = max_posterior * log_base ** (log2_matrix[i][j] - max_log2_value)
                else:
                    log2_matrix[i][j] = 0

    # return scale


def add_log2_probs(log2_X, log2_Y):

    # swap them if log2_Y is the bigger number
    if log2_X < log2_Y:
        log2_X, log2_Y = log2_Y, log2_X

    to_add = log2(1 + 2 ** (log2_Y - log2_X))

    if math.isnan(to_add) or math.isinf(to_add):
        return log2_X
    else:
        return log2_X + to_add


def add_log_probs(log_X, log_Y):

    # swap them if log2_Y is the bigger number
    if log_X < log_Y:
        log_X, log_Y = log_Y, log_X

    to_add = math.log(1 + math.exp(log_Y - log_X))

    if math.isnan(to_add) or math.isinf(to_add):
        return log_X
    else:
        return log_X + to_add


def normalize(array):
    total = sum(array)

    # protect from division by zeros
    if total == 0.0:
        return array

    return [float(a) / total for a in array]


def chunks(array, size):
    for i in xrange(0, len(array), size):
        yield array[i: i + size]


def open_file(fname, mode='r'):
    if fname.endswith('.gz'):
        if mode == 'r':
            return io.BufferedReader(gzip.open(fname))
        else:
            return io.BufferedWriter(gzip.open(fname, mode))
    else:
        return open(fname, mode)


def append_and_unlink(from_fname, to_fname):
    # append the file content of from_file to to_file and delete from_file

    with open_file(from_fname) as from_f, \
         open_file(to_fname, 'a') as to_f:
        to_f.write(from_f.read())

    os.unlink(from_fname)


def not_None(thing):
    return thing is not None


def sign(value):
    return 0 if value == 0 else -1 if value < 0 else 1


def parse(s, types, splitter='\t'):
    buf = s.split(splitter)
    if len(buf) != len(types):
        error('Cannot parse string: ' + s + ' with types: ' + str(types))
    return [t(b) for t, b in zip(types, buf)]


def median(array):
    so = sorted(array)
    n = len(so)
    if n % 2:
        return so[n / 2]
    else:
        return float(so[n / 2 - 1] + so[n / 2]) / 2

