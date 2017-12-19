# ChromTime: Modeling Spatio-temporal Dynamics of Chromatin Marks 

## Installation

ChromTime is written in Python 2.7 and C and can be run on Linux and MacOS. 

You need to install Numpy and SciPy before you can run ChromTime. You can use the following URL or your favorite package manager:

https://www.scipy.org/


After you installed Numpy and SciPy, you have to compile the C code with `gcc` or a similar C compiler. You can do this by simply typing `make` in
the ChromTime directory. If you are using a C compiler that is different from `gcc`, you should edit the provided Makefile accordingly.
After the compilation has finished, you should be able to run ChromTime by typing:

    $ python ChromTime.py
    usage: ChromTime.py [-h] [-a ALIGNED_FNAMES [ALIGNED_FNAMES ...]]
                        [-c CONTROL_FNAMES [CONTROL_FNAMES ...]] [-i ORDER_FNAME]
                        [-g GENOME] [-o DIRECTORY] [-b INT] [-t INT] [-s SHIFT]
                        [-q FDR_FOR_DECODING] [--q-value-seed Q_VALUE_SEED]
                        [--p-value-extend P_VALUE_EXTEND] [--min-gap MIN_GAP]
                        [-n INT] [-m FILE] [-d FILE] [-p PREFIX] [--skip-training]
                        [--merge-peaks] [--no-output-signal-files] [--broad]
                        [--atac]
    
    ChromTime: Modeling Spatio-temporal Dynamics of Chromatin Marks
    
    optional arguments:
      -h, --help            show this help message and exit
      -a ALIGNED_FNAMES [ALIGNED_FNAMES ...], --aligned-reads ALIGNED_FNAMES [ALIGNED_FNAMES ...]
                            Bed files with aligned reads for each time point in
                            the correct order
      -c CONTROL_FNAMES [CONTROL_FNAMES ...], --control-reads CONTROL_FNAMES [CONTROL_FNAMES ...]
                            Bed files with aligned reads for control (input) for
                            each time point in the correct order
      -i ORDER_FNAME, --input-order-file ORDER_FNAME
                            A file with paths to files with foreground and control
                            aligned reads - one line per time point, in the right
                            order.
      -g GENOME, --genome GENOME
                            Genome. One of: [hg18, hg19, mm10, mm9, zv9] or path
                            to a file with chromosome sizes one per line
      -o DIRECTORY, --output-dir DIRECTORY
                            Output directory
      -b INT, --bin-size INT
                            Bin size in base pairs (Default: 200)
      -t INT, --threads INT
                            Number of threads to use (Default: 1)
      -s SHIFT, --shift SHIFT
                            Number of bases to shift each read (Default: 100)
      -q FDR_FOR_DECODING, --q-value FDR_FOR_DECODING
                            False discovery rate (Q-value) for calling peaks at
                            each time point (Default: 0.05)
      --q-value-seed Q_VALUE_SEED
                            FDR threshold to call significant bins (Default: 0.05)
      --p-value-extend P_VALUE_EXTEND
                            FDR threshold to call significant bins (Default: 0.15)
      --min-gap MIN_GAP     Minimum gap between significant regions before they
                            are joined (Default: 600)
      -n INT, --n-training-examples INT
                            Number of training examples to use. (Default: 10000)
      -m FILE, --model-file FILE
                            Pickled model file to load a learned the model from
      -d FILE, --data-file FILE
                            Pickled data file to load
      -p PREFIX, --prefix PREFIX
                            prefix for the output files
      --skip-training       Skip EM training (False)
      --merge-peaks         Merge significant peaks across time points instead of
                            splitting them (False)
      --no-output-signal-files
                            Do not output signal files for each time point in
                            wiggle format (False)
      --broad               Use default settings for broad marks. Equivalent to
                            "-b 500 --min-gap 1500 --merge-peaks" (False)
    
## Input

Sample input data taken from [Zhang et al. 2012](http://www.cell.com/cell/fulltext/S0092-8674(12)00293-0)
for H3K4me2 on chr19 and the corresponding ChromTime output can be found in `example/`. The ChromTime output can be also viewed in the UCSC Genome Browser
if you click [here](https://genome.ucsc.edu/cgi-bin/hgTracks?hgS_doOtherUser=submit&hgS_otherUserName=pfiziev&hgS_otherUserSessionName=t_cell_development.H3K4me2.ChromTime).

ChromTime takes as input aligned reads from ChIP-seq experiments in [BED format](https://genome.ucsc.edu/FAQ/FAQformat.html#format1), 
one per time point. In addition, you can specify aligned reads from control experiments (Input, IgG, etc) to be used as background. 
There are two ways specify the input fo ChromTime:

1) Specify a space-separated list of files for each time point with the `-a` and `-c` options. The files should be in the right order according to the time course. For example:

        python ~/software/ChromTime/ChromTime.py -a data/FLDN1_H3K4me2.merged.chr19.bed.gz data/FLDN2a_H3K4me2.merged.chr19.bed.gz data/FLDN2b_H3K4me2.merged.chr19.bed.gz data/ThyDN3_H3K4me2.merged.chr19.bed.gz data/ThyDP_H3K4me2.merged.chr19.bed.gz -c data/FLDN1_Input_rep1.chr19.bed.gz data/FLDN2a_Input_rep1.chr19.bed.gz data/FLDN2b_Input_rep1.chr19.bed.gz data/ThyDN3_Input_rep1.chr19.bed.gz data/ThyDP_Input_rep1.chr19.bed.gz -o t_cell_development.H3K4ME2 -p t_cell_development.H3K4ME2 -t 4 -g mm9

2) Specify a tab-separated text file with the path to ChIP-seq and background reads for each time point with the `-i` option. 
The file names should come in the correct order according the time course (i.e. line 1 corresponds to time point 1, line 2 to time point 2 and so on)
Paths can be either relative or absolute. For example:
    
        python ~/software/ChromTime/ChromTime.py -i input_order -o t_cell_development.H3K4ME2 -p t_cell_development.H3K4ME2 -t 4 -g mm9

Where `input_order` contains

    data/FLDN1_H3K4me2.merged.chr19.bed.gz	data/FLDN1_Input_rep1.chr19.bed.gz
    data/FLDN2a_H3K4me2.merged.chr19.bed.gz	data/FLDN2a_Input_rep1.chr19.bed.gz
    data/FLDN2b_H3K4me2.merged.chr19.bed.gz	data/FLDN2b_Input_rep1.chr19.bed.gz
    data/ThyDN3_H3K4me2.merged.chr19.bed.gz	data/ThyDN3_Input_rep1.chr19.bed.gz
    data/ThyDP_H3K4me2.merged.chr19.bed.gz	data/ThyDP_Input_rep1.chr19.bed.gz
    

## Output

All output files will be stored in the directory specified by the `-o` option. Each file will be prefixed with the label given by the `-p` option, if it is specified. 
Here is an example output generated from the above order file:

#####Predictions for each time point
        
    t_cell_development.H3K4ME2.FLDN1_H3K4me2.merged.chr19.ChromTime_timepoint_predictions.bed.gz
    t_cell_development.H3K4ME2.FLDN2a_H3K4me2.merged.chr19.ChromTime_timepoint_predictions.bed.gz
    t_cell_development.H3K4ME2.FLDN2b_H3K4me2.merged.chr19.ChromTime_timepoint_predictions.bed.gz
    t_cell_development.H3K4ME2.ThyDN3_H3K4me2.merged.chr19.ChromTime_timepoint_predictions.bed.gz
    t_cell_development.H3K4ME2.ThyDP_H3K4me2.merged.chr19.ChromTime_timepoint_predictions.bed.gz

The `.ChromTime_timepoint_predictions.bed.gz` files contain the predicted boundaries in gzipped BED format for each peak at each time point.
They have the following format:

    chromosome <tab> predicted start <tab> predicted end <tab> region id # predicted class <tab> 1000 <tab> . <tab> color

For example:

    chr19	4865000	4865800	chr19-100-1#S-C-S-S/S-C-S-S	1000	.	4865000	4865800	90,90,90
    chr19	36907400	36911400	chr19-1000-1#S-S-S-S/S-S-C-S	1000	.	36907400	36911400	90,90,90
    chr19	36978600	36979400	chr19-1001-1#S-S-S-S/S-S-S-S	1000	.	36978600	36979400	90,90,90

    
#####Full output from ChromTime that contains extra information like posterior probabilities    
 
    t_cell_development.H3K4ME2.ChromTime_full_output.bed.gz 

#####Learned model and intermediate data stored in pickle format

    t_cell_development.H3K4ME2.data.pickle
    t_cell_development.H3K4ME2.model.pickle

#####Initial block boundaries and significant bins for each time point
    
    t_cell_development.H3K4ME2_FLDN1_H3K4me2.merged.chr19.block_boundaries.bed.gz
    t_cell_development.H3K4ME2_FLDN1_H3K4me2.merged.chr19.significant_bins.bed.gz
    t_cell_development.H3K4ME2_FLDN2a_H3K4me2.merged.chr19.significant_bins.bed.gz
    t_cell_development.H3K4ME2_FLDN2b_H3K4me2.merged.chr19.significant_bins.bed.gz
    t_cell_development.H3K4ME2_ThyDN3_H3K4me2.merged.chr19.significant_bins.bed.gz
    t_cell_development.H3K4ME2_ThyDP_H3K4me2.merged.chr19.significant_bins.bed.gz
    
#####Observed and expected read counts and RPKM values in wiggle format for each time point
        
    t_cell_development.H3K4ME2_FLDN1_H3K4me2.merged.chr19.READ_COUNTS.wig.gz
    t_cell_development.H3K4ME2_FLDN1_H3K4me2.merged.chr19.EXPECTED.wig.gz
    t_cell_development.H3K4ME2_FLDN1_H3K4me2.merged.chr19.RPKM.wig.gz

    t_cell_development.H3K4ME2_FLDN2a_H3K4me2.merged.chr19.READ_COUNTS.wig.gz
    t_cell_development.H3K4ME2_FLDN2a_H3K4me2.merged.chr19.EXPECTED.wig.gz
    t_cell_development.H3K4ME2_FLDN2a_H3K4me2.merged.chr19.RPKM.wig.gz

    t_cell_development.H3K4ME2_FLDN2b_H3K4me2.merged.chr19.READ_COUNTS.wig.gz
    t_cell_development.H3K4ME2_FLDN2b_H3K4me2.merged.chr19.EXPECTED.wig.gz
    t_cell_development.H3K4ME2_FLDN2b_H3K4me2.merged.chr19.RPKM.wig.gz

    t_cell_development.H3K4ME2_ThyDN3_H3K4me2.merged.chr19.READ_COUNTS.wig.gz
    t_cell_development.H3K4ME2_ThyDN3_H3K4me2.merged.chr19.EXPECTED.wig.gz
    t_cell_development.H3K4ME2_ThyDN3_H3K4me2.merged.chr19.RPKM.wig.gz

    t_cell_development.H3K4ME2_ThyDP_H3K4me2.merged.chr19.READ_COUNTS.wig.gz
    t_cell_development.H3K4ME2_ThyDP_H3K4me2.merged.chr19.EXPECTED.wig.gz
    t_cell_development.H3K4ME2_ThyDP_H3K4me2.merged.chr19.RPKM.wig.gz

#####ChromTime log file

    t_cell_development.H3K4ME2.log


