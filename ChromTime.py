import argparse
import pprint

import cPickle as pickle
import math

from utils import *
from scipy.stats import poisson
from constants import *
from call_boundary_dynamics import call_boundary_dynamics

import random

# You should always set the random seed in the beginning of your software
# in order to obtain reproducible results!
# Here, we set the random seed to 42.
# After extensive literature search, we concluded that a seed of 42 is optimal, because
# 42 is the answer to the Ultimate Question of Life, the Universe and Everything.
# [The Hitchhiker's Guide to the Galaxy, Douglas Adams]
random.seed(42)

GENOMES_DIR = os.path.join(ROOT_DIR, 'genomes')
MAX_REGION_LENGTH = 30


def read_chrom_lengths(fname, bin_size):
    chrom_lengths = {}
    with open(fname) as in_f:
        for line in in_f:
            chrom, size = line.strip().split()
            chrom_lengths[chrom] = int(size) / bin_size
    return chrom_lengths


def read_aligned_reads(reads_fname, shift, bin_size, chrom_lengths=None):

    read_counts = dict((c, [0] * chrom_lengths[c]) for c in chrom_lengths)

    total_reads = 0

    echo('Reading reads from:', reads_fname)
    skipped = 0
    skipped_chromosomes = set()

    with open_file(reads_fname) as in_f:
        for line in in_f:
            if line.startswith('#'):
                continue

            buf = line.strip().split()

            if len(buf) < 4:
                error('Incorrect format of input file:', reads_fname + '\nLine: ' + line +
                      'Aligned reads should be in BED format: \n'
                      '"chromosome\tstart\tend\tstrand" or "chromosome\tstart\tend\tname\tscore\tstrand"')

            chrom = buf[0]

            try:
                start = int(buf[1])
            except ValueError:
                error("Start coordinate should be integer:", line)

            try:
                end = int(buf[2])
            except ValueError:
                error("End coordinate should be integer:", line)

            if end < start:
                error('Start coordinate is greater than end coordinate for line:', line)

            if len(buf) == 4:
                strand = buf[3]
            else:
                strand = buf[5]

            if strand not in ['+', '-']:
                error('Strand should be one of [+, -]:', line)

            if strand == '+':
                read_start = (start + shift) / bin_size
            else:
                read_start = (end - shift) / bin_size

            if chrom not in read_counts or read_start < 0 or read_start >= len(read_counts[chrom]):

                if chrom not in read_counts:
                    skipped_chromosomes.add(chrom)

                skipped += 1
                continue

            read_counts[chrom][read_start] += 1
            total_reads += 1

    echo('Total reads used for peak calling:', total_reads)

    if skipped > 0:
        echo('WARNING: Skipped reads outside of chromosome boundaries:', skipped)

        if len(skipped_chromosomes) > 0:
            echo('WARNING: Input file contains reads from non-standard chromosomes, which will be skipped:',
                 str(sorted(skipped_chromosomes)) + '\nStandard chromosomes for this genome assembly are:',
                 sorted(chrom_lengths))

    if total_reads == 0:
        error(reads_fname, 'has no sequencing reads that map to standard chromosomes for this genome assembly. '
              'Please check the input file!')

    return read_counts, total_reads


def merge_intervals(intervals, min_gap=0):
    merged = []

    for peak_idx in xrange(len(intervals)):

        if peak_idx == 0:
            merged.append(intervals[peak_idx])
        else:
            prev_peak_start, prev_peak_end = merged[-1]
            cur_peak_start, cur_peak_end = intervals[peak_idx]

            if overlap(prev_peak_start, prev_peak_end, cur_peak_start, cur_peak_end) >= -(min_gap + 1):
                merged[-1][1] = max(cur_peak_end, prev_peak_end)
            else:
                merged.append([cur_peak_start, cur_peak_end])

    return merged


def call_peaks(foreground_read_counts, total_foreground_reads,
               background_read_counts, total_background_reads,
               bin_size,
               p_value_extend,
               q_value_seed,
               min_gap,
               min_expected_reads,
               use_broad_window_for_background=False):

    SHORT_WINDOW = max(1, 500 / bin_size)   # 1 kb / 2
    MEDIUM_WINDOW = max(1, 2500 / bin_size)  # 5 kb / 2
    LONG_WINDOW = max(1, 10000 / bin_size)  # 20 kb / 2

    if use_broad_window_for_background:
        background_read_counts = foreground_read_counts
        total_background_reads = total_foreground_reads
        LONG_WINDOW = max(1, 25000 / bin_size)  # 50 kb / 2

    pseudo_one_read = float(min_expected_reads * total_background_reads) / total_foreground_reads

    n_total_bins = sum(len(bins) for bins in foreground_read_counts.itervalues())

    mean_background_reads = float(total_background_reads) / n_total_bins

    expected_read_counts = dict((c, [0] * len(foreground_read_counts[c])) for c in foreground_read_counts)

    if total_background_reads == 0:
        echo('Using average reads per bin as expected:', total_foreground_reads / float(n_total_bins))

    peaks = {}
    poisson_cache = {}
    echo('Calling significant bins')
    for chrom in foreground_read_counts:
        peaks[chrom] = [0] * len(foreground_read_counts[chrom])

        short_window = sum(background_read_counts[chrom][:SHORT_WINDOW])
        short_window_length = SHORT_WINDOW

        medium_window = sum(background_read_counts[chrom][:MEDIUM_WINDOW])
        medium_window_length = MEDIUM_WINDOW

        long_window = sum(background_read_counts[chrom][:LONG_WINDOW])
        long_window_length = LONG_WINDOW

        for bin_idx in xrange(len(foreground_read_counts[chrom])):

            fgr_reads = foreground_read_counts[chrom][bin_idx]

            if bin_idx >= SHORT_WINDOW:
                short_window -= background_read_counts[chrom][bin_idx - SHORT_WINDOW]
            else:
                short_window_length += 1

            if bin_idx + SHORT_WINDOW < len(background_read_counts[chrom]):
                short_window += background_read_counts[chrom][bin_idx + SHORT_WINDOW]
            else:
                short_window_length -= 1

            if bin_idx >= MEDIUM_WINDOW:
                medium_window -= background_read_counts[chrom][bin_idx - MEDIUM_WINDOW]
            else:
                medium_window_length += 1

            if bin_idx + MEDIUM_WINDOW < len(background_read_counts[chrom]):
                medium_window += background_read_counts[chrom][bin_idx + MEDIUM_WINDOW]
            else:
                medium_window_length -= 1

            if bin_idx >= LONG_WINDOW:
                long_window -= background_read_counts[chrom][bin_idx - LONG_WINDOW]
            else:
                long_window_length += 1

            if bin_idx + LONG_WINDOW < len(background_read_counts[chrom]):
                long_window += background_read_counts[chrom][bin_idx + LONG_WINDOW]
            else:
                long_window_length -= 1

            if use_broad_window_for_background:
                bgr_reads = max(float(long_window) / long_window_length,
                                mean_background_reads,
                                pseudo_one_read
                                )
                expected_reads = total_foreground_reads * bgr_reads / float(total_background_reads)
            else:
                if total_background_reads > 0:

                    bgr_reads = max(float(short_window) / short_window_length,
                                    float(medium_window) / medium_window_length,
                                    float(long_window) / long_window_length,
                                    mean_background_reads
                                    ,pseudo_one_read
                                    )

                    expected_reads = total_foreground_reads * bgr_reads / float(total_background_reads)
                else:
                    expected_reads = max(1., total_foreground_reads / float(n_total_bins))

            # cache the Poisson test
            key = (fgr_reads - 1, expected_reads)
            if key not in poisson_cache:
                poisson_cache[key] = poisson.sf(fgr_reads - 1, mu=expected_reads)

            peaks[chrom][bin_idx] = poisson_cache[key]

            expected_read_counts[chrom][bin_idx] = expected_reads

    echo('Computing p-value threshold at FDR of', q_value_seed)
    sorted_p_values = sorted([p for chrom in peaks for p in peaks[chrom]])
    n = len(sorted_p_values)

    q_value_strong = None

    for i, p_value in enumerate(sorted_p_values):

        if float(n * p_value) / (i + 1) <= q_value_seed:
            q_value_strong = p_value

    echo('p-value threshold:', q_value_strong)

    if q_value_strong is None:
        echo('ERROR: No significant peaks are found for this time point!\n'
             'Please, check your data and consider removing this time point or '
             'relaxing the FDR threshold with the --q-value-seed option.')
        exit(1)

    merged_peaks = {}
    for chrom in peaks:

        chrom_peaks = peaks[chrom]

        peak_bins = []
        in_peak = False
        peak_start = None
        n_bins = len(peaks[chrom])

        for bin_idx in xrange(n_bins):
            is_significant = (chrom_peaks[bin_idx] <= q_value_strong)

            if not in_peak and is_significant:
                in_peak = True
                peak_start = bin_idx

            if (not is_significant or bin_idx == n_bins - 1) and in_peak:
                peak_bins.append([peak_start, bin_idx])
                in_peak = False

        for peak_idx in xrange(len(peak_bins)):
            peak_start, peak_end = peak_bins[peak_idx]
            boundary = peak_start
            while boundary >= 0 and chrom_peaks[boundary] <= p_value_extend:
                boundary -= 1

            peak_start = boundary + 1

            boundary = peak_end
            while boundary < n_bins and chrom_peaks[boundary] <= p_value_extend:
                boundary += 1

            peak_end = boundary
            peak_bins[peak_idx] = [peak_start, peak_end]

        merged_peaks[chrom] = merge_intervals(peak_bins, min_gap=min_gap)

    return merged_peaks, expected_read_counts


def new_block(block_id,
              chrom,
              block_start,
              block_end,
              foreground_read_counts,
              foreground_total_read_counts,
              expected_read_counts,
              bin_size,
              is_subpeak):

    n_timepoints = len(foreground_read_counts)
    n_covariates = 2

    block_length = block_end - block_start

    block = {FOREGROUND_SIGNAL: [t_fgr_read_counts[chrom][block_start: block_end]
                                 for t_fgr_read_counts in foreground_read_counts],

             BLOCK_COVARIATES: cube(n_timepoints, block_length, n_covariates, default=1.),

             CHROMOSOME: chrom,

             BLOCK_LENGTH: block_length,

             BLOCK_OFFSET: block_start * bin_size,
             BLOCK_END: block_end * bin_size,
             IS_SUBPEAK: is_subpeak,
             BLOCK_ID: block_id,
             SPLIT_POINT: -1}

    for t, t_expected_read_counts in enumerate(expected_read_counts):
        for pos, pos_exp_rc in enumerate(t_expected_read_counts[chrom][block_start: block_end]):
            block[BLOCK_COVARIATES][t][pos][0] = math.log(pos_exp_rc)

    # If the region is too long, it has to be split at a point that is contained
    # in the peaks at all time points. We are going to cache the top positions with the maximum
    # mean of normalized signal, so that later we can pick where to split.
    # Only the first position is used during the EM, but all positions are used by the Viterbi decoding to
    # find the best assignment.

    block[SPLIT_POINT] = list(sorted(range(block_length + 1),
                                     reverse=True,
                                     key=lambda p:
                                     reduce(lambda x, y: x + y,
                                            [block[FOREGROUND_SIGNAL][t][min(p, block_length - 1)] /
                                             expected_read_counts[t][chrom][block_start + min(p, block_length - 1)]
                                                for t in xrange(n_timepoints)], 0)))[:MAX_REGION_LENGTH]

    return block


def find_best_splits(chrom,
                     part_start,
                     part_end,
                     peaks_in_partition,
                     foreground_read_counts,
                     expected_read_counts,
                     n_timepoints):

    GAP = 0
    PEAK = 1
    FLANKING = 2

    tracks = matrix(n_timepoints, part_end - part_start, default=FLANKING)

    for t in xrange(n_timepoints):
        for peak_idx in xrange(len(peaks_in_partition[t])):
            peak_start, peak_end = peaks_in_partition[t][peak_idx]
            for j in xrange(peak_start, peak_end):
                tracks[t][j - part_start] = PEAK

            if peak_idx > 0:
                prev_end = peaks_in_partition[t][peak_idx - 1][1]
                for j in xrange(prev_end, peak_start):
                    tracks[t][j - part_start] = GAP

    merged_gaps = []
    gap_counts = [[t for t in xrange(n_timepoints) if tracks[t][pos] in [GAP, FLANKING]]
                  for pos in xrange(part_end - part_start)]

    gap_only_timepoints = [[t for t in xrange(n_timepoints) if tracks[t][pos] == GAP]
                                for pos in xrange(part_end - part_start)]
    gap_start = None

    for pos in xrange(1, len(gap_counts)):
        if len(gap_counts[pos]) > len(gap_counts[pos - 1]):
            gap_start = pos

        if len(gap_counts[pos]) < len(gap_counts[pos - 1]):
            if gap_start is not None and len(gap_only_timepoints[pos - 1]) > 0:
                merged_gaps.append([part_start + gap_start, part_start + pos, gap_only_timepoints[pos - 1]])
            gap_start = None

    part_splits = [part_start]

    for gap_start, gap_end, gap_timepoints in merged_gaps:
        # find the position with the minimum rescaled foreground signal
        split_position = min(range(gap_start, gap_end), key=lambda p: min(foreground_read_counts[t][chrom][p]
                                                                          / expected_read_counts[t][chrom][p]
                                                                          for t in set(gap_timepoints)))
        part_splits.append(split_position)

    part_splits.append(part_end)

    return part_splits


def get_block_boundaries(peaks,
                         foreground_read_counts,
                         foreground_total_read_counts,
                         expected_read_counts,
                         bin_size,
                         merge_peaks):
    blocks = {}

    n_timepoints = len(peaks)
    chromosomes = sorted(set(chrom for p in peaks for chrom in p))
    MIN_GAP_BETWEEN_PEAKS_AT_DIFFERENT_TIMEPOINT = 2
    for chrom in chromosomes:

        all_chrom_intervals = sorted([start, end] for p in peaks for start, end in p.get(chrom, []))

        # merge overlapping and touching intervals
        merged_peaks = merge_intervals(all_chrom_intervals, min_gap=MIN_GAP_BETWEEN_PEAKS_AT_DIFFERENT_TIMEPOINT)

        # extend the peaks by a window from both sides
        EXTEND_WINDOW = 5
        prev_end = None

        for i in xrange(len(merged_peaks)):
            cur_start = merged_peaks[i][0]

            if i == 0:
              merged_peaks[i][0] = max(0, merged_peaks[i][0] - EXTEND_WINDOW)

            elif i > 0:
                if prev_end + EXTEND_WINDOW > cur_start - EXTEND_WINDOW:
                    window = (cur_start - prev_end) / 2
                else:
                    window = EXTEND_WINDOW

                merged_peaks[i - 1][1] += window
                merged_peaks[i][0] -= window

            if i == len(merged_peaks) - 1:
                merged_peaks[i][1] = min(merged_peaks[i][1] + EXTEND_WINDOW,
                                         len(foreground_read_counts[0][chrom]))

            prev_end = merged_peaks[i][1]

        peak_idx = [0] * n_timepoints

        for part_no, (part_start, part_end) in enumerate(merged_peaks):
            peaks_in_partition = [[] for _ in xrange(n_timepoints)]

            for t in xrange(n_timepoints):
                t_chrom_peaks = peaks[t].get(chrom, [])

                first_peak = peak_idx[t]

                while peak_idx[t] < len(t_chrom_peaks) and overlap(part_start, part_end, *t_chrom_peaks[peak_idx[t]]) > 0:
                    peak_idx[t] += 1

                last_peak = peak_idx[t]

                if last_peak > first_peak:
                    peaks_in_partition[t].extend(t_chrom_peaks[first_peak:last_peak])

            if merge_peaks:
                partition_splits = [part_start, part_end]
            else:
                # ignore peaks shorter than 2 bins for splitting
                MIN_PEAK_LENGTH = 400 / bin_size
                for t in xrange(n_timepoints):
                    if len(peaks_in_partition[t]) > 1:
                        peaks_in_partition[t] = [p for p in peaks_in_partition[t] if p[1] - p[0] >= MIN_PEAK_LENGTH]

                partition_splits = find_best_splits(chrom,
                                                    part_start,
                                                    part_end,
                                                    peaks_in_partition,
                                                    foreground_read_counts,
                                                    expected_read_counts,
                                                    n_timepoints)

            split_block = len(partition_splits) > 2
            for split_idx in xrange(len(partition_splits) - 1):

                block_start = partition_splits[split_idx]
                block_end = partition_splits[split_idx + 1]

                if split_block:
                    block_id = chrom + '-' + str(part_no + 1) + '-sub_' + str(split_idx + 1)
                else:
                    block_id = chrom + '-' + str(part_no + 1) + '-' + str(split_idx + 1)

                blocks[block_id] = new_block(block_id,
                                             chrom,
                                             block_start,
                                             block_end,
                                             foreground_read_counts,
                                             foreground_total_read_counts,
                                             expected_read_counts,
                                             bin_size,
                                             split_block)

    return blocks


def determine_block_boundaries(aligned_fnames,
                               control_fnames,
                               shift,
                               bin_size,
                               n_threads,
                               p_value_extend,
                               q_value_seed,
                               merge_peaks,
                               min_gap,
                               out_prefix,
                               chrom_lengths,
                               output_signal_files,
                               min_expected_reads,
                               use_broad_window_for_background=False):
    peaks = []

    foreground_read_counts = []
    foreground_total_read_counts = []

    expected_read_counts = []

    if control_fnames is None or len(control_fnames) == 0:
        echo('No control reads')
        control_fnames = [None] * len(aligned_fnames)

    elif len(control_fnames) == 1 and len(aligned_fnames) > 1:
        echo('Using the same control for all ChIP-seq experiments')
        control_fnames = [control_fnames[0]] * len(aligned_fnames)

    if len(control_fnames) != len(aligned_fnames):
        echo('ERROR: Please, specify exactly one control file for each time point!')
        exit(1)

    bgr_cache = {}

    for t_foreground_reads_fname, t_background_reads_fname in zip(aligned_fnames, control_fnames):

        t_foreground_read_counts, t_total_foreground_reads = read_aligned_reads(t_foreground_reads_fname,
                                                                                shift,
                                                                                bin_size,
                                                                                chrom_lengths=chrom_lengths)

        if t_background_reads_fname is None:
            echo("No control reads for:", t_foreground_reads_fname)

            t_background_read_counts = dict((c, [0] * chrom_lengths[c]) for c in chrom_lengths)
            t_total_background_reads = 0

        else:

            if t_background_reads_fname not in bgr_cache:
                bgr_cache[t_background_reads_fname] = read_aligned_reads(t_background_reads_fname,
                                                                         0,
                                                                         bin_size,
                                                                         chrom_lengths=chrom_lengths)
            else:
                echo('Using:', t_background_reads_fname, 'for control')

            t_background_read_counts, t_total_background_reads = bgr_cache[t_background_reads_fname]

        t_peaks, t_expected_read_counts = call_peaks(t_foreground_read_counts, t_total_foreground_reads,
                                                     t_background_read_counts, t_total_background_reads,
                                                     bin_size,
                                                     p_value_extend=p_value_extend,
                                                     q_value_seed=q_value_seed,
                                                     min_gap=min_gap,
                                                     min_expected_reads=min_expected_reads,
                                                     use_broad_window_for_background=use_broad_window_for_background)

        peaks.append(t_peaks)

        foreground_read_counts.append(t_foreground_read_counts)
        foreground_total_read_counts.append(t_total_foreground_reads)

        expected_read_counts.append(t_expected_read_counts)

        if output_signal_files:
            echo('Writing down significant peaks')
            _temp_out_prefix = out_prefix + '_' + os.path.split(t_foreground_reads_fname)[1].replace('.bed', '').replace('.gz', '')

            with open_file(_temp_out_prefix + '.significant_bins.bed.gz', 'w') as out_f:
                for chrom in sorted(t_peaks):

                    for peak_idx, (peak_start, peak_end) in enumerate(t_peaks[chrom]):
                        out_f.write('\t'.join(map(str, [chrom,
                                                        peak_start * bin_size,
                                                        peak_end * bin_size,
                                                        chrom + '-' + str(peak_idx + 1)])) + '\n')

            expected_out_f = open_file(_temp_out_prefix + '.EXPECTED.wig.gz', 'w')

            with open_file(_temp_out_prefix + '.RPKM.wig.gz', 'w') as rpkm_out_f, \
                 open_file(_temp_out_prefix + '.READ_COUNTS.wig.gz', 'w') as read_counts_out_f:

                title = os.path.split(t_foreground_reads_fname)[1].replace('.bed', '').replace('.gz', '')
                rpkm_out_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title + ' RPKM', title + ' RPKM'))
                read_counts_out_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title + ' RC', title + ' RC'))

                expected_out_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title + ' EXPECTED RC', title + ' EXPECTED RC'))

                for chrom in sorted(t_foreground_read_counts):
                    rpkm_out_f.write('fixedStep\tchrom=%s\tstart=0\tstep=%d\tspan=%d\n' % (chrom, bin_size, bin_size))
                    read_counts_out_f.write('fixedStep\tchrom=%s\tstart=0\tstep=%d\tspan=%d\n' % (chrom, bin_size, bin_size))

                    expected_out_f.write('fixedStep\tchrom=%s\tstart=0\tstep=%d\tspan=%d\n' % (chrom, bin_size, bin_size))

                    for peak_idx, read_count in enumerate(t_foreground_read_counts[chrom]):
                        rpkm_out_f.write('%.2lf\n' % (10 ** 9 * float(read_count) / (bin_size * t_total_foreground_reads)))
                        read_counts_out_f.write('%d\n' % read_count)

                        expected_out_f.write('%.2lf\n' % t_expected_read_counts[chrom][peak_idx])

            expected_out_f.close()

    blocks = get_block_boundaries(peaks,
                                  foreground_read_counts,
                                  foreground_total_read_counts,
                                  expected_read_counts,
                                  bin_size,
                                  merge_peaks)

    if output_signal_files:
        with open_file(out_prefix + '_' + os.path.split(aligned_fnames[0])[1].replace('.bed', '').replace('.gz', '')
                                                + '.block_boundaries.bed.gz', 'w') as out_f:
            for block_id in sorted(blocks):
                b = blocks[block_id]
                out_f.write('\t'.join(map(str, [b[CHROMOSOME],
                                                b[BLOCK_OFFSET],
                                                b[BLOCK_OFFSET] + b[BLOCK_LENGTH] * bin_size,
                                                block_id
                                       ])) + '\n')

    return blocks


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ChromTime: Modeling Spatio-temporal Dynamics of Chromatin Marks')

    g1 = parser.add_argument_group('Input data from command line')
    g1.add_argument('-a',
                    '--aligned-reads',
                    dest='aligned_fnames',
                    nargs='+',
                    help='BED files with aligned reads for each time point in the correct order')

    g1.add_argument('-c',
                    '--control-reads',
                    dest='control_fnames',
                    nargs='+',
                    help='BED files with aligned reads for control (input) for each time point in the correct order')

    g2 = parser.add_argument_group('Input data from order file')
    g2.add_argument('-i',
                    '--input-order-file',
                    dest='order_fname',
                    help='A tab-separated file with paths to files with foreground and control aligned reads '
                         '- one line per time point, in the right order.')

    g3 = parser.add_argument_group('Options')
    g3.add_argument('-m',
                    '--mode',
                    dest='mode',
                    choices=['punctate', 'narrow', 'broad'],
                    default=None,
                    help='punctate: equivalent to \"-b 200 --min-gap 600 --min-dynamic-prior 0.05\", '
                         'narrow: equivalent to \"-b 200 --min-gap 600 --min-dynamic-prior 0\", '
                         'broad: equivalent to \"-b 500 --min-gap 1500 --merge-peaks --min-dynamic-prior 0\"')

    g3.add_argument("-g", "--genome", dest="genome",
                    help="Genome. One of: [%s] or path to a file with chromosome sizes one per line"
                       % ', '.join(fname.replace('.txt', '') for fname in os.listdir(GENOMES_DIR)))

    g3.add_argument("-o", "--output-dir", dest="out_dir",
                    help="Output directory", metavar="DIRECTORY")

    g3.add_argument("-p", "--prefix", dest="prefix",
                    help="prefix for the output files")

    g3.add_argument("-b", "--bin-size", type=int, dest="bin_size", default=200,
                    help="Bin size in base pairs (Default: %(default)s)", metavar="INT")

    g3.add_argument("-t", "--threads", type=int, dest="n_threads", default=1,
                    help="Number of threads to use (Default: %(default)s)",
                    metavar="INT")

    g3.add_argument('-s',
                    '--shift',
                    type=int,
                    dest='shift',
                    help='Number of bases to shift each read (Default: %(default)s)',
                    default=100)

    g3.add_argument('-q',
                    '--q-value',
                    type=float,
                    dest='fdr_for_decoding',
                    help='False discovery rate (Q-value) for calling peaks at each time point (Default: %(default)s)',
                    default=0.05)

    g3.add_argument('--q-value-seed',
                    type=float,
                    dest='q_value_seed',
                    help='FDR threshold to call significant bins (Default: %(default)s)',
                    default=0.05)

    g3.add_argument('--p-value-extend',
                    type=float,
                    dest='p_value_extend',
                    help='FDR threshold to call significant bins (Default: %(default)s)',
                    default=0.15)

    g3.add_argument('--min-expected-reads',
                    type=int,
                    dest='min_expected_reads',
                    help='Minimum expected reads per bin for the background component (Default: %(default)s)',
                    default=1)

    g3.add_argument('--min-gap',
                    type=int,
                    dest='min_gap',
                    help='Minimum gap between significant regions before they are joined (Default: %(default)s)',
                    default=600)


    g3.add_argument("--merge-peaks", action="store_true", dest="merge_peaks", default=False,
                    help="Merge significant peaks across time points instead of splitting them (Default: %(default)s)")

    g3.add_argument("--min-dynamic-prior",
                    type=float,
                    dest="min_dynamic_prior",
                    default=0.0,
                    help="Minimum prior probability for each dynamic at each time point (Default: %(default)s)")

    g3.add_argument("--model-file", dest="model_fname",
                    help="Pickled model to load",
                    metavar="FILE")

    g3.add_argument("--data-file", dest="data_fname",
                    help="Pickled data to load",
                    metavar="FILE")

    g3.add_argument("--skip-training", action="store_true", dest="skip_training", default=False,
                    help="Skip EM training (Default: %(default)s)")

    g3.add_argument("-n",
                    "--n-training-examples",
                    type=int,
                    dest="n_training_examples",
                    default=10000,
                    help="Number of training examples to use. (Default: %(default)s)",
                    metavar="INT")

    g3.add_argument("--output-signal-files", action="store_true", dest="output_signal_files", default=False,
                    help="Output signal files for each time point in wiggle format (Default: %(default)s)")

    # below are legacy options

    parser.add_argument("--broad", action="store_true", dest="broad", default=False,
                        # help="Use default settings for broad marks. "
                        #      "Equivalent to \"-b 500 --min-gap 1500 --merge-peaks\" (%(default)s)",
                        help=argparse.SUPPRESS)

    parser.add_argument("--output_empty_blocks", action="store_true", dest="output_empty_blocks", default=False,
                        help=argparse.SUPPRESS)

    parser.add_argument("--keep-fixed-priors", action="store_true", dest="keep_fixed_priors", default=False,
                        help=argparse.SUPPRESS)

    parser.add_argument("--use-broad-window-for-background",
                        action="store_true",
                        dest="use_broad_window_for_background",
                        default=False,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # if no options were given by the user, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    if args.broad or args.mode == 'broad':
        args.bin_size = 500
        args.min_gap = 1500
        args.merge_peaks = True
        args.min_dynamic_prior = 0

    elif args.mode == 'punctate':
        args.bin_size = 200
        args.min_gap = 600
        args.merge_peaks = False
        args.min_dynamic_prior = 0.05
    elif args.mode == 'narrow':
        args.bin_size = 200
        args.min_gap = 600
        args.merge_peaks = False
        args.min_dynamic_prior = 0


    # elif args.atac:
    #     args.bin_size = 50
    #     args.min_gap = 150
    #     args.merge_peaks = False
    #     args.shift = 5

    bin_size = args.bin_size
    min_gap = args.min_gap
    merge_peaks = args.merge_peaks

    if args.order_fname:
        echo('Reading order file:', args.order_fname)
        aligned_fnames = []
        control_fnames = []
        order_dir = os.path.split(args.order_fname)[0]

        with open(args.order_fname) as in_f:
            for line in in_f:
                if re.match(r"^\s*$", line):
                    continue

                if re.match(r"\s*#\s*genome\s*=", line):
                    args.genome = line.split("=")[1].strip()
                    continue

                buf = line.strip().split()

                if len(buf) not in [1, 2]:
                    echo('ERROR in input order file. Each line should have at most two tab separated files:', line)
                    exit(1)

                aligned_fnames.append(os.path.join(order_dir, buf[0]))

                if len(buf) == 2:
                    control_fnames.append(os.path.join(order_dir, buf[1]))

        if len(control_fnames) == 0:
            control_fnames = None
    else:
        aligned_fnames = args.aligned_fnames
        control_fnames = args.control_fnames

    if not os.path.exists(args.out_dir):
        echo('Output directory will be created:', args.out_dir)
        os.mkdir(args.out_dir)

    if args.prefix:
        out_prefix = os.path.join(args.out_dir, args.prefix)
    else:
        out_prefix = os.path.join(args.out_dir, os.path.split(aligned_fnames[0])[1].replace('.bed', '').replace('.gz', ''))

    open_log(out_prefix + '.log')

    genome = args.genome

    echo('Command line:', ' '.join(sys.argv), level=ECHO_TO_LOGFILE)
    echo('Options:\n', pprint.pformat(vars(args)))

    genome_chrom_lengths = dict((fname.replace('.txt', ''),
                                 read_chrom_lengths(os.path.join(GENOMES_DIR, fname),
                                                    bin_size)) for fname in os.listdir(GENOMES_DIR))

    if genome is None:
        echo('ERROR: Genome is not specified. Use the -g option!')
        exit(1)

    if genome in genome_chrom_lengths:
        chrom_lengths = genome_chrom_lengths[genome]
    else:
        chrom_lengths = read_chrom_lengths(genome, bin_size)

    if args.data_fname:
        echo('Loading data from:', args.data_fname)
        with open(args.data_fname) as in_f:
            blocks = pickle.load(in_f)
    else:
        echo('Estimating initial block boundaries')

        blocks = determine_block_boundaries(aligned_fnames=aligned_fnames,
                                            control_fnames=control_fnames,

                                            shift=args.shift,

                                            bin_size=bin_size,
                                            n_threads=args.n_threads,

                                            p_value_extend=args.p_value_extend,
                                            q_value_seed=args.q_value_seed,
                                            merge_peaks=merge_peaks,
                                            min_gap=min_gap / bin_size,
                                            out_prefix=out_prefix,
                                            chrom_lengths=chrom_lengths,
                                            output_signal_files=args.output_signal_files,
                                            min_expected_reads=args.min_expected_reads,
                                            use_broad_window_for_background=args.use_broad_window_for_background)

        with open(out_prefix + '.data.pickle', 'w') as outf:
            echo('Storing blocks in:', out_prefix + '.data.pickle')
            pickle.dump(blocks, outf, protocol=pickle.HIGHEST_PROTOCOL)

    echo('Calling boundary dynamics')

    call_boundary_dynamics(blocks,
                           bin_size=bin_size,

                           model_fname=args.model_fname,
                           n_threads=args.n_threads,
                           skip_training=args.skip_training,

                           n_training_examples=args.n_training_examples,
                           max_region_length=MAX_REGION_LENGTH,
                           aligned_fnames=aligned_fnames,
                           out_prefix=out_prefix,
                           fdr_for_decoding=args.fdr_for_decoding,
                           output_empty_blocks=args.output_empty_blocks,
                           update_priors=not args.keep_fixed_priors,
                           min_dynamic_prior=args.min_dynamic_prior,
                           ignore_decreasing_LL_error=True)

    close_log()
