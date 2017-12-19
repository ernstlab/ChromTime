#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
//#define ABS(a) (((a) >= 0) ? (a) : (-(a)))


#define TRUE 1
#define FALSE 0

#define E_UNDERFLOW 1
#define SUCCESS 0

typedef double ldouble;
//typedef long double ldouble;
#define INF HUGE_VAL

//typedef long double ldouble;
//#define INF HUGE_VALL

typedef int BOOL;
#define CACHE_NOT_FOUND INF

#define LEFT_BOUNDARY 0
#define RIGHT_BOUNDARY 1

#define FORWARD_DIRECTION 0
#define BACKWARD_DIRECTION 1

inline int ABS(int x) {
    return x >= 0 ? x : -x;
}

inline ldouble FABS(ldouble x) {
    return x >= 0 ? x : -x;
}

inline int sign(int x){
    return x / ABS(x);
}


ldouble nb_model(int signal, ldouble * covariates, ldouble delta, ldouble * betas, int n_covariates) {
        ldouble nb_mean = 0;
        for (int i = 0; i < n_covariates; i ++ ) {
            nb_mean += covariates[i] * betas[i];
        }
        nb_mean = exp(nb_mean);

        return (lgamma(signal + delta) - lgamma(signal + 1) - lgamma(delta) +
                delta * (log(delta / (nb_mean + delta))) +
                signal * (log(nb_mean / (nb_mean + delta)))
                );
}

ldouble poisson_pmf(int k, ldouble Lambda, BOOL return_log) {
    if (return_log) {
        return k * log(Lambda) - lgamma(k + 1.0) - Lambda;
    } else {
        return exp(k * log(Lambda) - lgamma(k + 1.0) - Lambda);
    }
}

ldouble normal_pdf(ldouble x, ldouble mu, ldouble stddev, BOOL return_log) {
    ldouble u = (x - mu) / stddev;
    if (return_log) {
        return -u * u / 2 - log(sqrt(2 * M_PI) * stddev);
    } else {
        return exp(-u * u / 2) / (sqrt(2 * M_PI) * stddev);
    }
}


ldouble LogNormal_pdf(ldouble x, ldouble mu, ldouble stddev, BOOL return_log) {
    ldouble u = (log(x) - mu) / stddev;
    if (return_log) {
        return -u * u / 2 - log(sqrt(2 * M_PI) * stddev * x);
    } else {
        return exp(-u * u / 2) / (sqrt(2 * M_PI) * stddev * x);
    }
}


ldouble LogNormal_pdf_corrected(ldouble x, ldouble mu, ldouble stddev, BOOL return_log) {
    return (erf((log(x + 1) - mu) / (sqrt(2) * stddev)) - erf((log(x) - mu) / (sqrt(2) * stddev))) / 2;
}




ldouble laplace_pdf(ldouble x, ldouble mu, ldouble b, BOOL return_log) {
//    ldouble var = stddev * stddev;
    ldouble u = FABS(x - mu) / b;
    if (return_log) {
        return -u - log(2 * b);
    } else {
        return exp(-u) / (2 * b);
    }
}

ldouble studentsT_pdf(ldouble x, ldouble df, ldouble stddev, BOOL return_log) {
    ldouble u = x / stddev;
    if (return_log) {
        return lgamma((df + 1) / 2) - log(1 + u * u / df) * (df + 1) / 2 - log(df * M_PI) / 2 - log(stddev) - lgamma(df / 2);
    } else {
        return tgamma((df + 1) / 2) * pow(1 + u * u / df, -(df + 1) / 2) / (sqrt(df * M_PI) * stddev * tgamma(df / 2));
    }
}


int normalize_posteriors(ldouble *log_posteriors, int n_clusters) {

    // first, find the maximum posterior
    ldouble max_log_posterior = log_posteriors[0];
    int i;

    for (i = 1; i < n_clusters; i++) {
        if (log_posteriors[i] > max_log_posterior) {
            max_log_posterior = log_posteriors[i];
        }
    }

    if (max_log_posterior == -INF) {
        return E_UNDERFLOW;
    }

    // calculate the ratios between all other posteriors and the maximum posterior
    ldouble total_posterior_ratios = 0;
    for (i = 0; i < n_clusters; i++) {
        total_posterior_ratios += exp(log_posteriors[i] - max_log_posterior);
    }

    // in the case of underflow, return
    if (total_posterior_ratios == 0.0 || isinf(1. / total_posterior_ratios)) {
        return E_UNDERFLOW;
    }

    // calculate the maximum posterior
    ldouble max_posterior = 1. / total_posterior_ratios;

    // rescale cluster posteriors to sum to one
    for (i = 0; i < n_clusters; i++) {
        log_posteriors[i] = max_posterior * exp(log_posteriors[i] - max_log_posterior);
    }

    return SUCCESS;
}

int convert_and_normalize_log_matrix(ldouble ** log_matrix, int n_rows, int n_columns) {

    int i, j;
    ldouble max_log_value = -INF;
    // find the maximum log value
    for (i = 0; i < n_rows; i ++ ) {
        for (j = 0; j < n_columns; j ++ ) {
            if (log_matrix[i][j] > max_log_value)  {
                max_log_value = log_matrix[i][j];
            }
        }
    }

    // compute the sum of x / max_value for all x in the matrix
    ldouble total_ratio = 0;
    for (i = 0; i < n_rows; i ++ ) {
        for (j = 0; j < n_columns; j ++ ) {
            if (log_matrix[i][j] > -INF)  {
                total_ratio += exp(log_matrix[i][j] - max_log_value);
            }
        }
    }

    if (total_ratio == 0 || isinf(1. / total_ratio)) {
        return E_UNDERFLOW;
    }

    // convert all matrix values to probabilities
    ldouble max_probability = 1. / total_ratio;
    for (i = 0; i < n_rows; i ++ ) {
        for (j = 0; j < n_columns; j ++ ) {
            if (log_matrix[i][j] > -INF)  {
                log_matrix[i][j] = max_probability * exp(log_matrix[i][j] - max_log_value);
            } else {
                log_matrix[i][j] = 0;
            }
        }
    }
    return SUCCESS;

}

int normalize_log_cube(ldouble ***log_cube, int Z, int n_rows, int n_columns) {

    int i, j, k;
    ldouble max_log_value = -INF;
    // find the maximum log value
    for (k = 0; k < Z; k ++ ) {
        for (i = 0; i < n_rows; i ++ ) {
            for (j = 0; j < n_columns; j ++ ) {
                if (log_cube[k][i][j] > max_log_value)  {
                    max_log_value = log_cube[k][i][j];
                }
            }
        }
    }
    // compute the sum of x / max_value for all x in the matrix
    ldouble total_ratio = 0;
    for (k = 0; k < Z; k ++ ) {
        for (i = 0; i < n_rows; i ++ ) {
            for (j = 0; j < n_columns; j ++ ) {
                if (log_cube[k][i][j] > -INF)  {
                    total_ratio += exp(log_cube[k][i][j] - max_log_value);
                }
            }
        }
    }
    if (total_ratio == 0 || isinf(1. / total_ratio)) {
        return E_UNDERFLOW;
    }

    // convert all matrix values to probabilities
    ldouble max_probability = 1. / total_ratio;

    for (k = 0; k < Z; k ++ ) {
        for (i = 0; i < n_rows; i ++ ) {
            for (j = 0; j < n_columns; j ++ ) {
                if (log_cube[k][i][j] > -INF)  {
                    log_cube[k][i][j] = max_probability * exp(log_cube[k][i][j] - max_log_value);
                } else {
                    log_cube[k][i][j] = 0;
                }
            }
        }
    }
    return SUCCESS;

}


#define N_NOISE_STDDEVS 1000
int min_with_noise(int * array, ldouble * noise_means, ldouble * noise_stddevs, int start, int end) {
    int m = array[start] + noise_means[start] - N_NOISE_STDDEVS * noise_stddevs[start];
    for (int i = start; i < end; i ++) {
        int value = array[i] + noise_means[i] - N_NOISE_STDDEVS * noise_stddevs[i];

        if (value < m) {
            m = value;
        }
    }
    return m;
}

int max_with_noise(int * array, ldouble * noise_means, ldouble * noise_stddevs, int start, int end) {
    int m = array[start] + noise_means[start] + N_NOISE_STDDEVS * noise_stddevs[start];
    for (int i = start; i < end; i ++) {
        int value = array[i] + noise_means[i] + N_NOISE_STDDEVS * noise_stddevs[i];

        if (value > m) {
            m = value;
        }
    }
    return m;
}

ldouble ** matrix(int n_rows, int n_columns) {
    ldouble **M = (ldouble **) calloc(n_rows, sizeof (ldouble *));
    if (M == NULL) { fprintf(stderr, "Cannot allocate memory for matrix M!"); exit(1); }

    for (int i = 0; i < n_rows; i ++) {
        M[i] = (ldouble *) calloc(n_columns, sizeof(ldouble));
        if (M[i] == NULL) { fprintf(stderr, "Cannot allocate memory for matrix M!"); exit(1); }

    }
    return M;
}

ldouble * new_array(int n) {
    ldouble *a = (ldouble *) calloc(n, sizeof (ldouble));
    if (a == NULL) { fprintf(stderr, "Cannot allocate memory for array a!"); exit(1); }
    return a;
}

int ** integer_matrix(int n_rows, int n_columns) {
    int **M = (int **) calloc(n_rows, sizeof (int *));
    if (M == NULL) { fprintf(stderr, "Cannot allocate memory for matrix M!"); exit(1); }
    for (int i = 0; i < n_rows; i ++) {
        M[i] = (int *) calloc(n_columns, sizeof(int));
        if (M[i] == NULL) { fprintf(stderr, "Cannot allocate memory for matrix M!"); exit(1); }
    }
    return M;
}

void free_matrix(ldouble **M, int n) {
    for (int i = 0; i < n; i ++) {
        free(M[i]);
    }
    free(M);
}

void free_integer_matrix(int **M, int n) {
    for (int i = 0; i < n; i ++) {
        free(M[i]);
    }
    free(M);
}


void set_array(ldouble *array, int n, ldouble value) {
    for (int i = 0; i < n; i ++) {
        array[i] = value;
    }
}


void set_matrix(ldouble **M, int n_rows, int n_cols, ldouble value) {
    for (int i = 0; i < n_rows; i ++) {
        for (int j = 0; j < n_cols; j ++) {
            M[i][j] = value;
        }
    }
}


void set_integer_matrix(int **M, int n_rows, int n_cols, int value) {
    for (int i = 0; i < n_rows; i ++) {
        for (int j = 0; j < n_cols; j ++) {
            M[i][j] = value;
        }
    }
}


void print_array(ldouble *array, int n){
    printf("[ ");
    for (int i = 0; i < n; i ++) {
        printf("%.16le", array[i]);
        if (i < n - 1) {
            printf(", ");
        }
    }
    printf(" ]\n");
}

void print_matrix(ldouble **M, int n_rows, int n_cols) {
    printf("[");
    for (int i = 0; i < n_rows; i ++) {
        printf("[");
        for (int j = 0; j < n_cols; j ++) {
            printf("%.16le", M[i][j]);
            if (j < n_cols - 1) { printf(", "); }
        }
        printf("]");
        if (i < n_rows - 1) { printf(",\n"); }

    }
    printf("]\n");
}

void print_matrix_int(int **M, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i ++) {
        for (int j = 0; j < n_cols; j ++) {
            printf("%d\t", M[i][j]);
        }
        printf("\n");
    }
}


ldouble *** cube(int Z, int n_rows, int n_columns) {

    ldouble ***C = (ldouble ***) calloc(Z, sizeof (ldouble **));
    if (C == NULL) { fprintf(stderr, "Cannot allocate memory for cube C!"); exit(1); }

    for (int z = 0; z < Z; z ++) {
        C[z] = matrix(n_rows, n_columns);
    }
    return C;
}

int *** integer_cube(int Z, int n_rows, int n_columns) {
    int ***C = (int ***) calloc(Z, sizeof (int **));
    if (C == NULL) { fprintf(stderr, "Cannot allocate memory for cube C!"); exit(1); }

    for (int z = 0; z < Z; z ++) {
        C[z] = integer_matrix(n_rows, n_columns);
    }
    return C;
}

void free_cube(ldouble ***C, int Z, int n) {
    for (int z = 0; z < Z; z ++){
        for (int i = 0; i < n; i ++) {
            free(C[z][i]);
        }
        free(C[z]);
    }
    free(C);
}

void free_integer_cube(int ***C, int Z, int n) {
    for (int z = 0; z < Z; z ++){
        for (int i = 0; i < n; i ++) {
            free(C[z][i]);
        }
        free(C[z]);
    }
    free(C);
}


void set_cube(ldouble ***C, int Z, int n_rows, int n_cols, ldouble value) {
    for (int z = 0; z < Z; z ++) {
        for (int i = 0; i < n_rows; i ++) {
            for (int j = 0; j < n_cols; j ++) {
                C[z][i][j] = value;
            }
        }
    }
}

void set_integer_cube(int ***C, int Z, int n_rows, int n_cols, int value) {
    for (int z = 0; z < Z; z ++) {
        for (int i = 0; i < n_rows; i ++) {
            for (int j = 0; j < n_cols; j ++) {
                C[z][i][j] = value;
            }
        }
    }
}

void print_cube(ldouble ***C, int Z, int n_rows, int n_cols){

    for (int z = 0; z < Z; z ++) {
        printf("z = %d\n", z);
        printf("[");
        for (int i = 0; i < n_rows; i ++) {
            printf("[");
            for (int j = 0; j < n_cols; j ++) {
                printf("%.16le", C[z][i][j]);
                if (j < n_cols - 1) {printf(", ");}
            }
            printf("]");
            if (i < n_rows - 1) {printf(",\n");}

        }
        printf("]\n\n");
    }
}

void print_cube_int(int ***C, int Z, int n_rows, int n_cols){
    for (int z = 0; z < Z; z ++) {
        printf("z = %d\n", z);
        for (int i = 0; i < n_rows; i ++) {
            for (int j = 0; j < n_cols; j ++) {
                printf("%d\t", C[z][i][j]);
            }
            printf("\n");
        }
    }
}

ldouble * ldouble_array(int length) {
    ldouble * array = (ldouble *) calloc (length, sizeof (ldouble));
    if (array == NULL) { fprintf(stderr, "Cannot allocate memory for ldouble array with length %d", length); exit(1); }
    for (int i = 0; i < length; i++) {
        array[i] = 0;
    }
    return array;
}


void matcopy(ldouble **from_matrix, ldouble **to_matrix, int n_rows, int n_columns) {
    for (int i = 0; i < n_rows; i ++) {
        for (int j = 0; j < n_columns; j ++) {
            to_matrix[i][j] = from_matrix[i][j];
        }
    }
}

void array_copy(ldouble *from, ldouble *to, int n) {
    for (int i = 0; i < n; i ++) {
        to[i] = from[i];
    }
}


ldouble add_log_probs(ldouble log_X, ldouble log_Y) {
    if (log_X == -INF) {
        return log_Y;
    } else if (log_Y == -INF) {
        return log_X;
    }

    // swap them if log_Y is the bigger number
    if (log_X < log_Y) {
        ldouble _tmp = log_X;
        log_X = log_Y;
        log_Y = _tmp;
        }

    ldouble to_add = log(1 + exp(log_Y - log_X));
    if (to_add == -INF || to_add == INF) {
        return log_X;
    } else {
        return log_X + to_add;
    }
}
//
//int main() {
//    for (long int x = 0; x < 2091259980; x++) {
//        //printf("%d %le\n", x, poisson_pmf(x, 5));
//        ldouble z = poisson_pmf(x % 1000, 50.);
//    }
//    return 0;
//}