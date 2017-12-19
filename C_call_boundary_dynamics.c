#include <stdio.h>
#include <stdlib.h>

#include "utils.c"

#define EXPAND 1
#define CONTRACT 2
#define STEADY 0

#define MISSING_OBSERVATION -1

ldouble *** dynamics_caches;
ldouble ** noise_cache;

ldouble *** log_dynamics_caches;
ldouble ** log_noise_cache;
ldouble STUDENTST = 0;
int LAPLACE_NOISE = 0;
int CACHE_MAX_DISTANCE = 0;
int CACHE_N_DYNAMICS = 0;
int NOISE_CACHE_MAX_DISTANCE = 0;

void init_caches(int n_timepoints_minus_one,
                 int max_distance,
                 int n_dynamics) {

    // Set the seed of the random number generator
    srand(42);

    CACHE_MAX_DISTANCE = max_distance;
    CACHE_N_DYNAMICS = n_dynamics;

    log_dynamics_caches = (ldouble ***) malloc (n_dynamics * sizeof (ldouble **));

    if (log_dynamics_caches == NULL) {
        fprintf(stderr, "Not enough memory for dynamics caches\n");
        exit(1);
    }

    for (int i = 0; i < n_dynamics; i ++) {
        log_dynamics_caches[i] = matrix(n_timepoints_minus_one, max_distance);
        set_matrix(log_dynamics_caches[i], n_timepoints_minus_one , max_distance, CACHE_NOT_FOUND);
    }

}

void free_caches(int n_timepoints_minus_one, int n_dynamics) {
    for (int i = 0; i < n_dynamics; i ++) {
        free_matrix(log_dynamics_caches[i], n_timepoints_minus_one);
    }
    free(log_dynamics_caches);
}


ldouble background_model(int signal, ldouble * covariates, ldouble delta, ldouble * betas, int n_covariates) {
    return nb_model(signal, covariates, delta, betas, n_covariates);
}


ldouble foreground_model(int signal, ldouble * covariates, ldouble delta, ldouble * betas, int n_covariates) {
    return nb_model(signal, covariates, delta, betas, n_covariates);
}


int calculate_signal_cache(int * block_fgr,
                           ldouble * block_covariates,

                           int block_length,

                           int n_timepoints,
                           int n_covariates,

                           ldouble * foreground_delta,
                           ldouble * foreground_betas,

                           ldouble * background_delta,
                           ldouble * background_betas,

                           ldouble *** emission_cache) {

    for (int t_idx = 0; t_idx < n_timepoints; t_idx ++ ) {
        ldouble ** cur_cache = emission_cache[t_idx];
        set_matrix(cur_cache, block_length + 1, block_length + 1, -INF);

        ldouble * t_covariates = block_covariates + t_idx * block_length * n_covariates;

        int * t_fgr_signal = block_fgr + t_idx * block_length;

        ldouble all_background_log_prob = 0;

        for (int pos_idx = 0; pos_idx < block_length; pos_idx ++ ) {
            all_background_log_prob += background_model(t_fgr_signal[pos_idx],
                                                        t_covariates + pos_idx * n_covariates,
                                                        background_delta[t_idx],
                                                        background_betas + t_idx * n_covariates,
                                                        n_covariates);
        }

        ldouble left_flanking_log_prob = 0;

        for (int start = 0; start < block_length + 1; start ++) {
            ldouble peak_log_prob = 0;

            if (start > 0) {
                left_flanking_log_prob += background_model(t_fgr_signal[start - 1],
                                                           t_covariates + (start - 1) * n_covariates,
                                                           background_delta[t_idx],
                                                           background_betas + t_idx * n_covariates,
                                                           n_covariates);
            }

            ldouble right_flanking_log_prob = all_background_log_prob - left_flanking_log_prob;

            for (int end = start; end < block_length + 1; end ++ ) {

                if (start == end) {
                    cur_cache[start][end] = all_background_log_prob;
                    continue;
                }

                peak_log_prob += foreground_model(t_fgr_signal[end - 1],
                                                  t_covariates + (end - 1) * n_covariates,
                                                  foreground_delta[t_idx],
                                                  foreground_betas + t_idx * n_covariates,
                                                  n_covariates);

                right_flanking_log_prob -= background_model(t_fgr_signal[end - 1],
                                                            t_covariates + (end - 1) * n_covariates,
                                                            background_delta[t_idx],
                                                            background_betas + t_idx * n_covariates,
                                                            n_covariates);

                cur_cache[start][end] = left_flanking_log_prob + peak_log_prob + right_flanking_log_prob;
            }
        }
    }
    return SUCCESS;
}


int calculate_signal_cache_split(  int * block_fgr,
                                   ldouble * block_covariates,

                                   int block_length,

                                   int n_timepoints,
                                   int n_covariates,

                                   ldouble * foreground_delta,
                                   ldouble * foreground_betas,

                                   ldouble * background_delta,
                                   ldouble * background_betas,

                                   ldouble ** emission_cache) {

    for (int t_idx = 0; t_idx < n_timepoints; t_idx ++ ) {

        ldouble * cur_cache = emission_cache[t_idx];
        set_array(cur_cache, block_length + 1, -INF);

        ldouble * t_covariates = block_covariates + t_idx * block_length * n_covariates;

        int * t_fgr_signal = block_fgr + t_idx * block_length;

        ldouble all_background_log_prob = 0;

        for (int pos_idx = 0; pos_idx < block_length; pos_idx ++ ) {
            ldouble log_prob = background_model(t_fgr_signal[pos_idx],
                                                t_covariates + pos_idx * n_covariates,
                                                background_delta[t_idx],
                                                background_betas + t_idx * n_covariates,
                                                n_covariates);
            all_background_log_prob += log_prob;
        }

        ldouble right_flanking_log_prob = all_background_log_prob;
        ldouble peak_log_prob = 0;

        for (int end = 0; end < block_length + 1; end ++) {

            if (end > 0) {
                right_flanking_log_prob -= background_model(t_fgr_signal[end - 1],
                                                            t_covariates + (end - 1) * n_covariates,
                                                            background_delta[t_idx],
                                                            background_betas + t_idx * n_covariates,
                                                            n_covariates);

                peak_log_prob += foreground_model(t_fgr_signal[end - 1],
                                                  t_covariates + (end - 1) * n_covariates,
                                                  foreground_delta[t_idx],
                                                  foreground_betas + t_idx * n_covariates,
                                                  n_covariates);

            }

            cur_cache[end] = peak_log_prob + right_flanking_log_prob;
        }
    }
    return SUCCESS;
}


ldouble boundary_movement_model(int diff, int timepoint, ldouble *dynamics_params, ldouble *priors, int n_timepoints) {

    ldouble covariates[1] = {1.};

    ldouble move_prob;
    int dyn;

    if (diff == 0) {
        move_prob = 0;
        dyn = STEADY;
    } else {
        if (diff > 0) {
            dyn = EXPAND;

        } else {
            dyn = CONTRACT;
        }

        ldouble *t_params = dynamics_params + dyn * 2 * (n_timepoints - 1) + 2 * timepoint;

        int abs_diff = ABS(diff) - 1;

        if (log_dynamics_caches[dyn][timepoint][abs_diff] == CACHE_NOT_FOUND) {
            log_dynamics_caches[dyn][timepoint][abs_diff] = nb_model(abs_diff, covariates, t_params[0], t_params + 1, 1);
        }

        move_prob = log_dynamics_caches[dyn][timepoint][abs_diff];

    }

    ldouble prior = priors[dyn * (n_timepoints - 1) + timepoint];

    if (prior > 0) {
        return move_prob + log(prior);
    } else {
        return -INF;
    }
}



int forward(ldouble *** emission_cache,

            int n_timepoints,
            int block_length,

            ldouble * dynamics_params,

            ldouble * priors,

            ldouble *** F

            ) {

    matcopy(emission_cache[0], F[0], block_length + 1, block_length + 1);

    for (int t = 1; t < n_timepoints; t ++ ) {
        set_matrix(F[t], block_length + 1, block_length + 1, -INF);
        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {
            for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {

                ldouble gamma = -INF;

                for (int prev_start = 0; prev_start < block_length + 1; prev_start ++ ) {

                    for (int prev_end = prev_start; prev_end < block_length + 1; prev_end ++ ) {


                        ldouble log_prob = F[t - 1][prev_start][prev_end] +
                                           boundary_movement_model(prev_start - cur_start, t - 1, dynamics_params, priors, n_timepoints) +
                                           boundary_movement_model(cur_end - prev_end, t - 1, dynamics_params, priors, n_timepoints);

                        gamma = add_log_probs(gamma, log_prob);
                    }
                }
                gamma += emission_cache[t][cur_start][cur_end];
                F[t][cur_start][cur_end] = gamma;
            }
        }
    }
    return SUCCESS;
}


int backward(ldouble *** emission_cache,

            int n_timepoints,
            int block_length,

            ldouble * dynamics_params,

            ldouble * priors,

            ldouble *** B) {

    for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {
        for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {
            B[n_timepoints - 1][cur_start][cur_end] = 0;
        }
    }

    for (int t = n_timepoints - 2; t >= 0; t -- ) {
        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {
            for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {

                ldouble gamma = -INF;

                for (int next_start = 0; next_start < block_length + 1; next_start ++ ) {
                    for (int next_end = next_start; next_end < block_length + 1; next_end ++ ) {

                        ldouble log_prob = B[t + 1][next_start][next_end] +
                                           emission_cache[t + 1][next_start][next_end] +
                                           boundary_movement_model(cur_start - next_start, t, dynamics_params, priors, n_timepoints) +
                                           boundary_movement_model(next_end - cur_end, t, dynamics_params, priors, n_timepoints);

                        gamma = add_log_probs(gamma, log_prob);
                    }
                }
                B[t][cur_start][cur_end] = gamma;
            }
        }
    }
    return SUCCESS;
}


int forward_split(ldouble ** emission_cache,

                  int n_timepoints,
                  int block_length,

                  ldouble * dynamics_params,

                  ldouble * priors,

                  ldouble ** F) {

    array_copy(emission_cache[0], F[0], block_length + 1);

    for (int t = 1; t < n_timepoints; t ++ ) {

        for (int cur_end = 0; cur_end < block_length + 1; cur_end ++ ) {

            ldouble gamma = -INF;

            for (int prev_end = 0; prev_end < block_length + 1; prev_end ++ ) {

                ldouble log_prob = F[t - 1][prev_end] +
                                   boundary_movement_model(cur_end - prev_end, t - 1, dynamics_params, priors, n_timepoints);

                gamma = add_log_probs(gamma, log_prob);
            }

            gamma += emission_cache[t][cur_end];
            F[t][cur_end] = gamma;
        }
    }

    return SUCCESS;
}


int backward_split( ldouble ** emission_cache,

                    int n_timepoints,
                    int block_length,

                    ldouble * dynamics_params,

                    ldouble * priors,

                    ldouble ** B) {

    for (int cur_end = 0; cur_end < block_length + 1; cur_end ++ ) {
        B[n_timepoints - 1][cur_end] = 0;
    }

    for (int t = n_timepoints - 2; t >= 0; t -- ) {

        for (int cur_end = 0; cur_end < block_length + 1; cur_end ++ ) {

            ldouble gamma = -INF;

            for (int next_end = 0; next_end < block_length + 1; next_end ++ ) {

                ldouble log_prob = B[t + 1][next_end] +
                                   emission_cache[t + 1][next_end] +
                                   boundary_movement_model(next_end - cur_end, t, dynamics_params, priors, n_timepoints);

                gamma = add_log_probs(gamma, log_prob);
            }

            B[t][cur_end] = gamma;
        }

    }
    return SUCCESS;
}


int EM_step (int * block_fgr,
             ldouble * block_covariates,
             int block_length,

             int n_covariates,
             int n_timepoints,

             ldouble * foreground_delta,
             ldouble * foreground_betas,

             ldouble * background_delta,
             ldouble * background_betas,

             ldouble * dynamics_params,
             int n_dynamics,
             int n_dynamics_params,

             ldouble * priors,

             ldouble * param_info,

             ldouble * log_likelihood) {

    ldouble ** dynamics_posteriors = matrix(2, n_dynamics);
    if (dynamics_posteriors == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: dynamics_posteriors"); exit(1); }

    ldouble ** position_posteriors = matrix(block_length + 1, block_length + 1);
    if (position_posteriors == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: position_posteriors"); exit(1); }

    ldouble *** dist_posteriors = cube(2, n_dynamics, block_length + 1);
    if (dist_posteriors == NULL) { fprintf(stderr, "Cannot allocate memory for cube: dist_posteriors"); exit(1); }

    int status = SUCCESS;

    // the noise info should be at the beginning of the param_info
    ldouble * peak_posteriors = param_info;
    ldouble * total_posteriors_per_dynamic = param_info + block_length * n_timepoints;
    ldouble * dynamics_params_info_offset = total_posteriors_per_dynamic + n_dynamics * (n_timepoints - 1);

    // calculate the forward and backward matrices for each cluster
    ldouble *** F = cube(n_timepoints, block_length + 1, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for cube: F"); exit(1); }

    ldouble *** B = cube(n_timepoints, block_length + 1, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for cube: B"); exit(1); }

    ldouble *** emission_cache = cube(n_timepoints, block_length + 1, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for cube: emission_cache"); exit(1); }


    status = calculate_signal_cache(block_fgr,
                                    block_covariates,
                                    block_length,
                                    n_timepoints,
                                    n_covariates,
                                    foreground_delta,
                                    foreground_betas,
                                    background_delta,
                                    background_betas,
                                    emission_cache);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward(emission_cache,
                         n_timepoints,
                         block_length,
                         dynamics_params,
                         priors,
                         F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward(emission_cache,
                              n_timepoints,
                              block_length,
                              dynamics_params,
                              priors,
                              B);
    }

    // compute the log likelihood of the block

    *log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        for (int j = i; j < block_length + 1; j ++ ) {
            *log_likelihood = add_log_probs(*log_likelihood, F[n_timepoints - 1][i][j]);
        }
    }

    for (int t = 0; t < n_timepoints - 1; t ++ ) {
        set_matrix(position_posteriors, block_length + 1, block_length + 1, -INF);

        set_matrix(dynamics_posteriors, 2, n_dynamics, -INF);

        set_cube(dist_posteriors, 2, n_dynamics, block_length + 1, -INF);

        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {

            for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {

                for (int next_start = 0; next_start < block_length + 1; next_start ++ ) {

                    int start_dist = cur_start - next_start;

                    for (int next_end = next_start; next_end < block_length + 1; next_end ++ ) {

                        int end_dist = next_end - cur_end;

                        ldouble log_prob = F[t][cur_start][cur_end] +
                                   boundary_movement_model(start_dist, t, dynamics_params, priors, n_timepoints) +
                                   boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                                   B[t + 1][next_start][next_end] +
                                   emission_cache[t + 1][next_start][next_end];

                        position_posteriors[cur_start][cur_end] = add_log_probs(position_posteriors[cur_start][cur_end],
                                                                                log_prob);

                        for (int boundary_idx = 0; boundary_idx < 2; boundary_idx ++ ) {
                            int dist = boundary_idx == 0 ? start_dist : end_dist;
                            int dynamic = STEADY;
                            if (dist < 0) {
                                dynamic = CONTRACT;
                            } else if (dist > 0) {
                                dynamic = EXPAND;
                            }
                            int abs_dist = ABS(dist);

                            dist_posteriors[boundary_idx][dynamic][abs_dist] = add_log_probs(dist_posteriors[boundary_idx][dynamic][abs_dist],
                                                                                              log_prob);

                            dynamics_posteriors[boundary_idx][dynamic] = add_log_probs(dynamics_posteriors[boundary_idx][dynamic],
                                                                                      log_prob);
                        }
                    }
                }
            }
        }

        for (int boundary_idx = 0; boundary_idx < 2; boundary_idx ++ ) {
            status = normalize_posteriors(dynamics_posteriors[boundary_idx], n_dynamics);
            if (status != SUCCESS) {
                goto CLEANUP;
            }

            // store the posteriors for each dynamic on each side
            for (int dyn_idx = 0; dyn_idx < n_dynamics; dyn_idx ++ ) {
                *(total_posteriors_per_dynamic + dyn_idx * (n_timepoints - 1) + t) += dynamics_posteriors[boundary_idx][dyn_idx];

                status = normalize_posteriors(dist_posteriors[boundary_idx][dyn_idx], block_length + 1);
                if (status != SUCCESS) {
                    goto CLEANUP;
                }

                if (dyn_idx > 0) {

                    ldouble * dynamics_info = dynamics_params_info_offset + dyn_idx * (block_length + 1) * (n_timepoints - 1) + (block_length + 1) * t;

                    for (int dist = 1; dist < block_length + 1; dist ++ ) {

                        ldouble weight = dist_posteriors[boundary_idx][dyn_idx][dist] * dynamics_posteriors[boundary_idx][dyn_idx];
                        dynamics_info[dist] += weight;

                    }
                }
            }
        }

        status = convert_and_normalize_log_matrix(position_posteriors, block_length + 1, block_length + 1);
        if (status != SUCCESS) {
            goto CLEANUP;
        }

        for (int start = 0; start < block_length + 1; start ++ ) {
            for (int end = start; end < block_length + 1; end ++ ) {

                ldouble weight = position_posteriors[start][end];

                for (int pos = start; pos < end; pos ++ ) {
                    *(peak_posteriors + t * block_length + pos) += weight;
                }
            }
        }
    }

    int t = n_timepoints - 1;
    matcopy(F[t], position_posteriors, block_length + 1, block_length + 1);

    status = convert_and_normalize_log_matrix(position_posteriors, block_length + 1, block_length + 1);
    if (status != SUCCESS) {
        goto CLEANUP;
    }

    for (int start = 0; start < block_length + 1; start ++ ) {
        for (int end = start; end < block_length + 1; end ++ ) {

            ldouble weight = position_posteriors[start][end];

            for (int pos = start; pos < end; pos ++ ) {
                *(peak_posteriors + t * block_length + pos) += weight;
            }
        }
    }

    CLEANUP:

    free_cube(F, n_timepoints, block_length + 1);
    free_cube(B, n_timepoints, block_length + 1);
    free_cube(emission_cache, n_timepoints, block_length + 1);

    free_matrix(dynamics_posteriors, 2);
    free_matrix(position_posteriors, block_length + 1);
    free_cube(dist_posteriors, 2, n_dynamics);

    return status;
}


int EM_step_split (  int * block_fgr,
                     ldouble * block_covariates,
                     int block_length,

                     int n_covariates,
                     int n_timepoints,

                     ldouble * foreground_delta,
                     ldouble * foreground_betas,

                     ldouble * background_delta,
                     ldouble * background_betas,

                     ldouble * dynamics_params,
                     int n_dynamics,
                     int n_dynamics_params,

                     ldouble * priors,

                     ldouble * param_info,

                     ldouble * log_likelihood) {

    ldouble * dynamics_posteriors = new_array(n_dynamics);

    ldouble * position_posteriors = new_array(block_length + 1);

    ldouble ** dist_posteriors = matrix(n_dynamics, block_length + 1);
    if (dist_posteriors == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: dist_posteriors"); exit(1); }

    int status = SUCCESS;

    // the noise info should be at the beginning of the param_info
    ldouble * peak_posteriors = param_info;
    ldouble * total_posteriors_per_dynamic = param_info + block_length * n_timepoints;
    ldouble * dynamics_params_info_offset = total_posteriors_per_dynamic + n_dynamics * (n_timepoints - 1);

    // calculate the forward and backward matrices for each cluster
    ldouble ** F = matrix(n_timepoints, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: F"); exit(1); }

    ldouble ** B = matrix(n_timepoints, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: B"); exit(1); }

    ldouble ** emission_cache = matrix(n_timepoints, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: emission_cache"); exit(1); }


    status = calculate_signal_cache_split(  block_fgr,
                                            block_covariates,
                                            block_length,
                                            n_timepoints,
                                            n_covariates,
                                            foreground_delta,
                                            foreground_betas,
                                            background_delta,
                                            background_betas,
                                            emission_cache);
    if (status != SUCCESS) {
        goto CLEANUP;
    }
    int fwd_status = -1, bck_status = -1;

    fwd_status = forward_split(  emission_cache,
                                 n_timepoints,
                                 block_length,
                                 dynamics_params,
                                 priors,
                                 F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward_split(  emission_cache,
                                      n_timepoints,
                                      block_length,
                                      dynamics_params,
                                      priors,
                                      B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }

    // compute the log likelihood of the block
    *log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        *log_likelihood = add_log_probs(*log_likelihood, F[n_timepoints - 1][i]);
    }

    for (int t = 0; t < n_timepoints - 1; t ++ ) {

        set_array(position_posteriors, block_length + 1, -INF);
        set_array(dynamics_posteriors, n_dynamics, -INF);
        set_matrix(dist_posteriors, n_dynamics, block_length + 1, -INF);

        for (int cur_end = 0; cur_end < block_length + 1; cur_end ++ ) {

            for (int next_end = 0; next_end < block_length + 1; next_end ++ ) {

                int end_dist = next_end - cur_end;

                ldouble log_prob = F[t][cur_end] +
                           boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                           B[t + 1][next_end] +
                           emission_cache[t + 1][next_end];

                position_posteriors[cur_end] = add_log_probs(position_posteriors[cur_end], log_prob);

                int dynamic = STEADY;
                if (end_dist < 0) {
                    dynamic = CONTRACT;
                } else if (end_dist > 0) {
                    dynamic = EXPAND;
                }
                int abs_dist = ABS(end_dist);

                dist_posteriors[dynamic][abs_dist] = add_log_probs(dist_posteriors[dynamic][abs_dist], log_prob);
                dynamics_posteriors[dynamic] = add_log_probs(dynamics_posteriors[dynamic], log_prob);
            }
        }

        status = normalize_posteriors(dynamics_posteriors, n_dynamics);
        if (status != SUCCESS) {
            goto CLEANUP;
        }

        // store the posteriors for each dynamic on each side
        for (int dyn_idx = 0; dyn_idx < n_dynamics; dyn_idx ++ ) {
            *(total_posteriors_per_dynamic + dyn_idx * (n_timepoints - 1) + t) += dynamics_posteriors[dyn_idx];

            status = normalize_posteriors(dist_posteriors[dyn_idx], block_length + 1);
            if (status != SUCCESS) {
                goto CLEANUP;
            }

            if (dyn_idx > 0) {

                ldouble * dynamics_info = dynamics_params_info_offset + dyn_idx * (block_length + 1) * (n_timepoints - 1) + (block_length + 1) * t;

                for (int dist = 1; dist < block_length + 1; dist ++ ) {

                    ldouble weight = dist_posteriors[dyn_idx][dist] * dynamics_posteriors[dyn_idx];
                    dynamics_info[dist] = weight;

                }
            }
        }

        status = normalize_posteriors(position_posteriors, block_length + 1);
        if (status != SUCCESS) {
            goto CLEANUP;
        }

        for (int end = 0; end < block_length + 1; end ++ ) {

            ldouble weight = position_posteriors[end];

            for (int pos = 0; pos < end; pos ++ ) {
                *(peak_posteriors + t * block_length + pos) += weight;
            }
        }
    }

    int t = n_timepoints - 1;
    array_copy(F[t], position_posteriors, block_length + 1);

    status = normalize_posteriors(position_posteriors, block_length + 1);
    if (status != SUCCESS) {
        goto CLEANUP;
    }

    for (int end = 0; end < block_length + 1; end ++ ) {

        ldouble weight = position_posteriors[end];

        for (int pos = 0; pos < end; pos ++ ) {
            *(peak_posteriors + t * block_length + pos) += weight;
        }
    }

    CLEANUP:

    free_matrix(F, n_timepoints);
    free_matrix(B, n_timepoints);
    free_matrix(emission_cache, n_timepoints);

    free(dynamics_posteriors);
    free(position_posteriors);
    free_matrix(dist_posteriors, n_dynamics);

    return status;
}


int calculate_posteriors (int * block_fgr,
             ldouble * block_covariates,
             int block_length,

             int n_covariates,
             int n_timepoints,

             ldouble * foreground_delta,
             ldouble * foreground_betas,

             ldouble * background_delta,
             ldouble * background_betas,

             ldouble * dynamics_params,
             int n_dynamics,

             ldouble * priors,

             ldouble * dynamics_posteriors) {

    int status = SUCCESS;

    // calculate the forward and backward matrices for each cluster
    ldouble *** F = cube(n_timepoints, block_length + 1, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for cube: F"); exit(1); }

    ldouble *** B = cube(n_timepoints, block_length + 1, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for cube: B"); exit(1); }

    ldouble *** emission_cache = cube(n_timepoints, block_length + 1, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for cube: emission_cache"); exit(1); }

    status = calculate_signal_cache(block_fgr,
                                    block_covariates,
                                    block_length,
                                    n_timepoints,
                                    n_covariates,
                                    foreground_delta,
                                    foreground_betas,
                                    background_delta,
                                    background_betas,
                                    emission_cache);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward(emission_cache,
                         n_timepoints,
                         block_length,
                         dynamics_params,
                         priors,
                         F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward(emission_cache,
                              n_timepoints,
                              block_length,
                              dynamics_params,
                              priors,
                              B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }

    for (int t = 0; t < n_timepoints - 1; t ++ ) {
        ldouble * t_dynamics_posteriors = dynamics_posteriors + t * 2 * n_dynamics;

        for (int b_side = 0; b_side < 2; b_side ++ ) {
            for (int dyn_idx = 0; dyn_idx < n_dynamics; dyn_idx ++ ) {
                * (t_dynamics_posteriors + b_side * n_dynamics + dyn_idx) = -INF;
            }
        }

        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {

            for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {

                for (int next_start = 0; next_start < block_length + 1; next_start ++ ) {

                    int start_dist = cur_start - next_start;

                    for (int next_end = next_start; next_end < block_length + 1; next_end ++ ) {

                        int end_dist = next_end - cur_end;

                        ldouble log_prob = F[t][cur_start][cur_end] +
                                   boundary_movement_model(start_dist, t, dynamics_params, priors, n_timepoints) +
                                   boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                                   B[t + 1][next_start][next_end] +
                                   emission_cache[t + 1][next_start][next_end];

                        for (int boundary_idx = 0; boundary_idx < 2; boundary_idx ++ ) {
                            int dist = (boundary_idx == 0) ? start_dist : end_dist;
                            int dynamic = STEADY;
                            if (dist < 0) {
                                dynamic = CONTRACT;
                            } else if (dist > 0) {
                                dynamic = EXPAND;
                            }

                            * (t_dynamics_posteriors + boundary_idx * n_dynamics + dynamic) = add_log_probs(* (t_dynamics_posteriors + boundary_idx * n_dynamics + dynamic),
                                                                                      log_prob);
                        }
                    }
                }
            }
        }

        for (int boundary_idx = 0; boundary_idx < 2; boundary_idx ++ ) {

            status = normalize_posteriors(t_dynamics_posteriors + boundary_idx * n_dynamics, n_dynamics);
            if (status != SUCCESS) {
                goto CLEANUP;
            }
        }
    }

    CLEANUP:

    free_cube(F, n_timepoints, block_length + 1);
    free_cube(B, n_timepoints, block_length + 1);
    free_cube(emission_cache, n_timepoints, block_length + 1);

    return status;
}


int calculate_posteriors_split ( int * block_fgr,
                                 ldouble * block_covariates,
                                 int block_length,

                                 int n_covariates,
                                 int n_timepoints,

                                 ldouble * foreground_delta,
                                 ldouble * foreground_betas,

                                 ldouble * background_delta,
                                 ldouble * background_betas,

                                 ldouble * dynamics_params,
                                 int n_dynamics,

                                 ldouble * priors,

                                 ldouble * dynamics_posteriors,
                                 ldouble * total_log_likelihood) {

    int status = SUCCESS;

    // calculate the forward and backward matrices for each cluster
    ldouble ** F = matrix(n_timepoints, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: F"); exit(1); }

    ldouble ** B = matrix(n_timepoints, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: B"); exit(1); }

    ldouble ** emission_cache = matrix(n_timepoints, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: emission_cache"); exit(1); }

    status = calculate_signal_cache_split(block_fgr,
                                          block_covariates,
                                          block_length,
                                          n_timepoints,
                                          n_covariates,
                                          foreground_delta,
                                          foreground_betas,
                                          background_delta,
                                          background_betas,
                                          emission_cache);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward_split(emission_cache,
                               n_timepoints,
                               block_length,
                               dynamics_params,
                               priors,
                               F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward_split(emission_cache,
                                    n_timepoints,
                                    block_length,
                                    dynamics_params,
                                    priors,
                                    B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }


    *total_log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        *total_log_likelihood = add_log_probs(*total_log_likelihood, F[n_timepoints - 1][i]);
    }


    for (int t = 0; t < n_timepoints - 1; t ++ ) {
        ldouble * t_dynamics_posteriors = dynamics_posteriors + t * n_dynamics;

        for (int dyn_idx = 0; dyn_idx < n_dynamics; dyn_idx ++ ) {
            * (t_dynamics_posteriors + dyn_idx) = -INF;
        }

        for (int cur_end = 0; cur_end < block_length + 1; cur_end ++ ) {

            for (int next_end = 0; next_end < block_length + 1; next_end ++) {

                int end_dist = next_end - cur_end;

                ldouble log_prob = F[t][cur_end] +
                                   boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                                   B[t + 1][next_end] +
                                   emission_cache[t + 1][next_end];

                int dynamic = STEADY;
                if (end_dist < 0) {
                    dynamic = CONTRACT;
                } else if (end_dist > 0) {
                    dynamic = EXPAND;
                }

                * (t_dynamics_posteriors + dynamic) = add_log_probs(* (t_dynamics_posteriors + dynamic), log_prob);

            }

        }

        status = normalize_posteriors(t_dynamics_posteriors, n_dynamics);
        if (status != SUCCESS) {
            goto CLEANUP;
        }
    }

    CLEANUP:

    free_matrix(F, n_timepoints);
    free_matrix(B, n_timepoints);
    free_matrix(emission_cache, n_timepoints);

    return status;
}


int compute_Viterbi_path(int * block_fgr,
                         ldouble * block_covariates,
                         int block_length,
                         int block_offset,

                         int n_covariates,
                         int n_timepoints,

                         ldouble * foreground_delta,
                         ldouble * foreground_betas,

                         ldouble * background_delta,
                         ldouble * background_betas,

                         ldouble * dynamics_params,
                         int n_dynamics,

                         ldouble * priors,

                         int bin_size,

                         ldouble * total_log_likelihood,
                         ldouble * combo_likelihood,
                         int * trajectories,
                         int * positions,
                         int * time_frame,
                         int * is_peak_timepoint) {

    ldouble *** emission_cache = cube(n_timepoints, block_length + 1, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for cube: emission_cache"); exit(1); }


    calculate_signal_cache(block_fgr,
                            block_covariates,
                            block_length,
                            n_timepoints,
                            n_covariates,
                            foreground_delta,
                            foreground_betas,
                            background_delta,
                            background_betas,
                            emission_cache);


    ldouble *** DP = cube(n_timepoints, block_length + 1, block_length + 1);

    if (DP == NULL) { fprintf(stderr, "Cannot allocate memory for cube: DP"); exit(1); }
    set_cube(DP, n_timepoints, block_length + 1, block_length + 1, -INF);

    ldouble *** trace_start = cube(n_timepoints, block_length + 1, block_length + 1);
    if (trace_start == NULL) { fprintf(stderr, "Cannot allocate memory for cube: trace_start"); exit(1); }
    set_cube(trace_start, n_timepoints, block_length + 1, block_length + 1, -INF);

    ldouble *** trace_end = cube(n_timepoints, block_length + 1, block_length + 1);
    if (trace_end == NULL) { fprintf(stderr, "Cannot allocate memory for cube: trace_end"); exit(1); }
    set_cube(trace_end, n_timepoints, block_length + 1, block_length + 1, -INF);

    ldouble * start_ties = new_array(block_length + 1);
    ldouble * end_ties = new_array(block_length + 1);

    int n_ties = 0;
    int random_tie = -1;

    matcopy(emission_cache[0], DP[0], block_length + 1, block_length + 1);

    *total_log_likelihood = -INF;

    for (int t = 1; t < n_timepoints; t ++ ) {

        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {

            for (int cur_end = cur_start; cur_end < (is_peak_timepoint[t] ? (block_length + 1) : (cur_start + 1)); cur_end ++ ) {
                n_ties = 0;

                ldouble max_log_prob = -INF;
                ldouble total_log_prob = -INF;

                for (int prev_start = 0; prev_start < block_length + 1; prev_start ++ ) {
                    for (int prev_end = prev_start; prev_end < (is_peak_timepoint[t - 1] ? (block_length + 1) : (prev_start + 1)); prev_end ++ ) {

                        ldouble log_prob = DP[t - 1][prev_start][prev_end] +
                                   boundary_movement_model(prev_start - cur_start, t - 1, dynamics_params, priors, n_timepoints) +
                                   boundary_movement_model(cur_end - prev_end, t - 1, dynamics_params, priors, n_timepoints);

                        total_log_prob = add_log_probs(total_log_prob, log_prob);

                        if (log_prob > max_log_prob) {
                            max_log_prob = log_prob;
                            n_ties = 0;

                        }

                        if (log_prob == max_log_prob) {
                            start_ties[n_ties] = prev_start;
                            end_ties[n_ties] = prev_end;
                            n_ties ++;
                        }

                    }
                }

                max_log_prob += emission_cache[t][cur_start][cur_end];

                *total_log_likelihood = add_log_probs(*total_log_likelihood,
                                                      total_log_prob + emission_cache[t][cur_start][cur_end]);

                DP[t][cur_start][cur_end] = max_log_prob;

                random_tie = rand() % n_ties;

                trace_start[t][cur_start][cur_end] = start_ties[random_tie];
                trace_end[t][cur_start][cur_end] = end_ties[random_tie];
            }
        }
    }

    int best_start = -1;
    int best_end = -1;
    n_ties = 0;

    ldouble best_score = -INF;

    for (int s = 0; s < block_length + 1; s ++ ) {
        for (int e = s; e < (is_peak_timepoint[n_timepoints - 1] ? (block_length + 1) : (s + 1)); e ++ ) {
            if (DP[n_timepoints - 1][s][e] > best_score) {
                best_score = DP[n_timepoints - 1][s][e];
                n_ties = 0;
            }

           if (DP[n_timepoints - 1][s][e] == best_score) {
                start_ties[n_ties] = s;
                end_ties[n_ties] = e;
                n_ties ++;
            }
        }
    }

    random_tie = rand() % n_ties;
    best_start = start_ties[random_tie];
    best_end = end_ties[random_tie];

    * combo_likelihood = DP[n_timepoints - 1][best_start][best_end];

    *(positions + n_timepoints - 1) = block_offset + best_start * bin_size;
    *(positions + 2 * n_timepoints - 1) = block_offset + best_end * bin_size;

    for (int t = n_timepoints - 1; t >= 1; t -- ) {

        int prev_best_start = trace_start[t][best_start][best_end];
        int prev_best_end = trace_end[t][best_start][best_end];

        *(positions + t - 1) = block_offset + prev_best_start * bin_size;
        *(positions + n_timepoints + t - 1) = block_offset + prev_best_end * bin_size;

        *(trajectories + t - 1) = (prev_best_start == best_start) ? STEADY : (prev_best_start > best_start) ? EXPAND : CONTRACT;
        *(trajectories + (n_timepoints - 1) + t - 1) = (prev_best_end == best_end) ? STEADY : (prev_best_end < best_end) ? EXPAND : CONTRACT;

        best_start = prev_best_start;
        best_end = prev_best_end;

    }

    // determine the first and the last time points

    int first_timepoint = -1;
    int last_timepoint = -1;

    for (int t = 0; t < n_timepoints; t ++ ) {
        if (*(positions + t) != *(positions + n_timepoints + t)) {
            if (first_timepoint == -1) {
                first_timepoint = t;
            }
            last_timepoint = t;
        }
    }
    * time_frame = first_timepoint;
    * (time_frame + 1) = last_timepoint;

    free_cube(emission_cache, n_timepoints, block_length + 1);
    free_cube(DP, n_timepoints, block_length + 1);
    free_cube(trace_start, n_timepoints, block_length + 1);
    free_cube(trace_end, n_timepoints, block_length + 1);
    free(start_ties);
    free(end_ties);

    return SUCCESS;

}


int compute_Viterbi_path_split(int * block_fgr,
                               ldouble * block_covariates,
                               int side,

                               int block_length,
                               int block_offset,

                               int n_covariates,
                               int n_timepoints,

                               ldouble * foreground_delta,
                               ldouble * foreground_betas,

                               ldouble * background_delta,
                               ldouble * background_betas,

                               ldouble * dynamics_params,
                               int n_dynamics,

                               ldouble * priors,

                               int bin_size,

                               ldouble * total_log_likelihood,
                               ldouble * combo_likelihood,
                               int * trajectories,
                               int * positions,
                               int * time_frame,
                               int * is_peak_timepoint) {

    ldouble ** emission_cache = matrix(n_timepoints, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: emission_cache"); exit(1); }

    calculate_signal_cache_split(block_fgr,
                                 block_covariates,
                                 block_length,
                                 n_timepoints,
                                 n_covariates,
                                 foreground_delta,
                                 foreground_betas,
                                 background_delta,
                                 background_betas,
                                 emission_cache);

    ldouble ** DP = matrix(n_timepoints, block_length + 1);

    if (DP == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: DP"); exit(1); }
    set_matrix(DP, n_timepoints, block_length + 1, -INF);

    ldouble ** trace_end = matrix(n_timepoints, block_length + 1);
    if (trace_end == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: trace_end"); exit(1); }
    set_matrix(trace_end, n_timepoints, block_length + 1, -INF);

    array_copy(emission_cache[0], DP[0], block_length + 1);

    *total_log_likelihood = -INF;

    for (int t = 1; t < n_timepoints; t ++ ) {

        for (int cur_end = 0; cur_end < (is_peak_timepoint[t] ? (block_length + 1) : 1); cur_end ++ ) {

            ldouble max_log_prob = -INF;
            ldouble total_log_prob = -INF;

            int best_prev_end = -1;

            for (int prev_end = 0; prev_end < (is_peak_timepoint[t - 1] ? (block_length + 1) : 1); prev_end ++ ) {

                    ldouble log_prob = DP[t - 1][prev_end] +
                                       boundary_movement_model(cur_end - prev_end,
                                                               t - 1,
                                                               dynamics_params,
                                                               priors,
                                                               n_timepoints);

                    total_log_prob = add_log_probs(total_log_prob, log_prob);

                    if (log_prob >= max_log_prob) {
                        best_prev_end = prev_end;
                        max_log_prob = log_prob;
                    }
            }

            max_log_prob += emission_cache[t][cur_end];

            *total_log_likelihood = add_log_probs(*total_log_likelihood,
                                                  total_log_prob + emission_cache[t][cur_end]);

            DP[t][cur_end] = max_log_prob;

            trace_end[t][cur_end] = best_prev_end;
        }
    }

    int best_end = -1;
    ldouble best_score = -INF;

    if (!is_peak_timepoint[n_timepoints - 1]) {
        best_score = DP[n_timepoints - 1][0];
        best_end = 0;
    } else {
        for (int e = 0; e < block_length + 1; e ++ ) {
            if (DP[n_timepoints - 1][e] >= best_score) {
                best_score = DP[n_timepoints - 1][e];
                best_end = e;
            }
        }
    }

    * combo_likelihood = DP[n_timepoints - 1][best_end];

    *(positions + n_timepoints - 1) = block_offset + side * best_end * bin_size;

    for (int t = n_timepoints - 1; t >= 1; t -- ) {

        int prev_best_end = trace_end[t][best_end];

        *(positions + t - 1) = block_offset + side * prev_best_end * bin_size;

        *(trajectories + t - 1) = (prev_best_end == best_end) ? STEADY : (prev_best_end < best_end) ? EXPAND : CONTRACT;

        best_end = prev_best_end;

    }

    // determine the first and the last time points

    int first_timepoint = -1;
    int last_timepoint = -1;

    for (int t = 0; t < n_timepoints; t ++ ) {
        if (*(positions + t) != block_offset) {
            if (first_timepoint == -1) {
                first_timepoint = t;
            }
            last_timepoint = t;
        }
    }

    * time_frame = first_timepoint;
    * (time_frame + 1) = last_timepoint;

    free_matrix(emission_cache, n_timepoints);
    free_matrix(DP, n_timepoints);
    free_matrix(trace_end, n_timepoints);

    return SUCCESS;

}


int compute_posterior_paths(int * block_fgr,
                            ldouble * block_covariates,
                            int block_length,
                            int block_offset,

                            int n_covariates,
                            int n_timepoints,

                            ldouble * foreground_delta,
                            ldouble * foreground_betas,

                            ldouble * background_delta,
                            ldouble * background_betas,

                            ldouble * dynamics_params,
                            int n_dynamics,

                            ldouble * priors,

                            int bin_size,

                            ldouble * total_log_likelihood,
                            ldouble * combo_likelihood,
                            int * trajectories,
                            int * positions,
                            int * time_frame,
                            int * is_peak_timepoint // this is not doing anything!!
                            ) {

    int status = SUCCESS;

    // calculate the forward and backward matrices for each cluster
    ldouble *** F = cube(n_timepoints, block_length + 1, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for cube: F"); exit(1); }

    ldouble *** B = cube(n_timepoints, block_length + 1, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for cube: B"); exit(1); }

    ldouble *** emission_cache = cube(n_timepoints, block_length + 1, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for cube: emission_cache"); exit(1); }

    ldouble ** peak_probs = matrix(n_timepoints, 2);
    if (peak_probs == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: peak_probs"); exit(1); }
    set_matrix(peak_probs, n_timepoints, 2, -INF);

    ldouble *** DP = cube(n_timepoints, block_length + 1, block_length + 1);

    if (DP == NULL) { fprintf(stderr, "Cannot allocate memory for cube: DP"); exit(1); }
    set_cube(DP, n_timepoints, block_length + 1, block_length + 1, -INF);

    calculate_signal_cache(block_fgr,
                            block_covariates,
                            block_length,
                            n_timepoints,
                            n_covariates,
                            foreground_delta,
                            foreground_betas,
                            background_delta,
                            background_betas,
                            emission_cache);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward(emission_cache,
                         n_timepoints,
                         block_length,
                         dynamics_params,
                         priors,
                         F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward(emission_cache,
                              n_timepoints,
                              block_length,
                              dynamics_params,
                              priors,
                              B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }

    matcopy(emission_cache[0], DP[0], block_length + 1, block_length + 1);

    *total_log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        for (int j = i; j < block_length + 1; j ++ ) {
            *total_log_likelihood = add_log_probs(*total_log_likelihood, F[n_timepoints - 1][i][j]);
        }
    }


    for (int t = 0; t < n_timepoints - 1; t ++ ) {

        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {

            for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {

                ldouble total_log_prob = -INF;

                for (int next_start = 0; next_start < block_length + 1; next_start ++ ) {

                    int start_dist = cur_start - next_start;

                    for (int next_end = next_start; next_end < block_length + 1; next_end ++ ) {

                        int end_dist = next_end - cur_end;

                        ldouble log_prob = F[t][cur_start][cur_end] +
                                           boundary_movement_model(start_dist, t, dynamics_params, priors, n_timepoints) +
                                           boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                                           B[t + 1][next_start][next_end] +
                                           emission_cache[t + 1][next_start][next_end];

                        total_log_prob = add_log_probs(total_log_prob, log_prob);

                        if (t == n_timepoints - 2) {
                            DP[t + 1][next_start][next_end] = add_log_probs(DP[t + 1][next_start][next_end], log_prob);
                            if (next_start == next_end) {
                                peak_probs[t + 1][0] = add_log_probs(peak_probs[t + 1][0], log_prob);
                            } else {
                                peak_probs[t + 1][1] = add_log_probs(peak_probs[t + 1][1], log_prob);
                            }
                        }
                    }
                }

                DP[t][cur_start][cur_end] = total_log_prob;

                if (cur_start == cur_end) {
                    peak_probs[t][0] = add_log_probs(peak_probs[t][0], total_log_prob);
                } else {
                    peak_probs[t][1] = add_log_probs(peak_probs[t][1], total_log_prob);
                }
            }
        }
    }

    int best_start = -1;
    int best_end = -1;

    int prev_best_start = -1;
    int prev_best_end = -1;

    for (int t = 0; t < n_timepoints; t ++ ) {
        status = normalize_posteriors(peak_probs[t], 2);

        if (status != SUCCESS) {
            goto CLEANUP;
        }

        ldouble best_score = -INF;

        if (peak_probs[t][0] >= peak_probs[t][1]) {

            for (int p = 0; p < block_length + 1; p ++ ) {
                if (DP[t][p][p] > best_score) {
                    best_score = DP[t][p][p];
                    best_start = p;
                    best_end = p;
                }
            }
        } else {
            for (int s = 0; s < block_length + 1; s ++ ) {
                for (int e = s; e < block_length + 1; e ++ ) {
                    if (DP[t][s][e] > best_score) {
                        best_score = DP[t][s][e];
                        best_start = s;
                        best_end = e;
                    }
                }
            }
        }

        *(positions + t) = block_offset + best_start * bin_size;
        *(positions + n_timepoints + t) = block_offset + best_end * bin_size;

        if (t > 0) {
            *(trajectories + t - 1) = (prev_best_start == best_start) ? STEADY : (prev_best_start > best_start) ? EXPAND : CONTRACT;
            *(trajectories + (n_timepoints - 1) + t - 1) = (prev_best_end == best_end) ? STEADY : (prev_best_end < best_end) ? EXPAND : CONTRACT;

        }

        * combo_likelihood += DP[t][best_start][best_end];

        prev_best_start = best_start;
        prev_best_end = best_end;

    }

    // determine the first and the last time points

    int first_timepoint = -1;
    int last_timepoint = -1;

    for (int t = 0; t < n_timepoints; t ++ ) {
        if (*(positions + t) != *(positions + n_timepoints + t)) {
            if (first_timepoint == -1) {
                first_timepoint = t;
            }
            last_timepoint = t;
        }
    }
    * time_frame = first_timepoint;
    * (time_frame + 1) = last_timepoint;


    CLEANUP:

    free_cube(emission_cache, n_timepoints, block_length + 1);
    free_cube(DP, n_timepoints, block_length + 1);
    free_cube(F, n_timepoints, block_length + 1);
    free_cube(B, n_timepoints, block_length + 1);

    free_matrix(peak_probs, n_timepoints);

    return status;

}


int compute_posterior_path_split( int * block_fgr,
                                  ldouble * block_covariates,
                                  int side,
                                  int block_length,
                                  int block_offset,

                                  int n_covariates,
                                  int n_timepoints,

                                  ldouble * foreground_delta,
                                  ldouble * foreground_betas,

                                  ldouble * background_delta,
                                  ldouble * background_betas,

                                  ldouble * dynamics_params,
                                  int n_dynamics,

                                  ldouble * priors,

                                  int bin_size,

                                  ldouble * total_log_likelihood,
                                  ldouble * combo_likelihood,
                                  int * trajectories,
                                  int * positions,
                                  int * time_frame,
                                  int * is_peak_timepoint // this is not doing anything!!
                                  ) {


    int status = SUCCESS;

    // calculate the forward and backward matrices for each cluster
    ldouble * position_posteriors = new_array(block_length + 1);

    ldouble ** F = matrix(n_timepoints, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: F"); exit(1); }

    ldouble ** B = matrix(n_timepoints, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: B"); exit(1); }

    ldouble ** emission_cache = matrix(n_timepoints, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: emission_cache"); exit(1); }

    calculate_signal_cache_split(block_fgr,
                                 block_covariates,
                                 block_length,
                                 n_timepoints,
                                 n_covariates,
                                 foreground_delta,
                                 foreground_betas,
                                 background_delta,
                                 background_betas,
                                 emission_cache);

    ldouble ** DP = matrix(n_timepoints, block_length + 1);

    if (DP == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: DP"); exit(1); }
    set_matrix(DP, n_timepoints, block_length + 1, -INF);

    array_copy(emission_cache[0], DP[0], block_length + 1);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward_split(emission_cache,
                               n_timepoints,
                               block_length,
                               dynamics_params,
                               priors,
                               F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward_split(emission_cache,
                                    n_timepoints,
                                    block_length,
                                    dynamics_params,
                                    priors,
                                    B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }

    *total_log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        *total_log_likelihood = add_log_probs(*total_log_likelihood, F[n_timepoints - 1][i]);
    }

    for (int t = 0; t < n_timepoints - 1; t ++ ) {

        for (int cur_end = 0; cur_end < block_length + 1; cur_end ++ ) {

            ldouble total_log_prob = -INF;

            for (int next_end = 0; next_end < block_length + 1; next_end ++ ) {

                int end_dist = next_end - cur_end;

                ldouble log_prob = F[t][cur_end] +
                                   boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                                   B[t + 1][next_end] +
                                   emission_cache[t + 1][next_end];

                total_log_prob = add_log_probs(total_log_prob, log_prob);

                if (t == n_timepoints - 2) {
                    DP[t + 1][next_end] = add_log_probs(DP[t + 1][next_end], log_prob);
                }
            }

            DP[t][cur_end] = total_log_prob;
        }
    }

    int best_end = -1;

    int prev_best_end = -1;

    for (int t = 0; t < n_timepoints; t ++ ) {
        array_copy(DP[t], position_posteriors, block_length + 1);
        status = normalize_posteriors(position_posteriors, block_length + 1);
        if (status != SUCCESS) {
            goto CLEANUP;
        }

        ldouble best_score = -INF;

        for (int e = 0; e < block_length + 1; e ++ ) {
            if (DP[t][e] > best_score) {
                best_score = DP[t][e];
                best_end = e;
            }
        }

        *(positions + t) = block_offset + side * best_end * bin_size;

        if (t > 0) {
            *(trajectories + t - 1) = (prev_best_end == best_end) ? STEADY : (prev_best_end < best_end) ? EXPAND : CONTRACT;
        }

        * combo_likelihood += DP[t][best_end];

        prev_best_end = best_end;

    }

    // determine the first and the last time points

    int first_timepoint = -1;
    int last_timepoint = -1;

    for (int t = 0; t < n_timepoints; t ++ ) {
        if (*(positions + t) != block_offset) {
            if (first_timepoint == -1) {
                first_timepoint = t;
            }
            last_timepoint = t;
        }
    }
    * time_frame = first_timepoint;
    * (time_frame + 1) = last_timepoint;


    CLEANUP:

    free_matrix(emission_cache, n_timepoints);
    free_matrix(DP, n_timepoints);
    free_matrix(F, n_timepoints);
    free_matrix(B, n_timepoints);
    free(position_posteriors);
    return status;


}


int get_no_peak_likelihoods_split(int * block_fgr,
                                  ldouble * block_covariates,

                                  int block_length,

                                  int n_covariates,
                                  int n_timepoints,

                                  ldouble * foreground_delta,
                                  ldouble * foreground_betas,

                                  ldouble * background_delta,
                                  ldouble * background_betas,

                                  ldouble * dynamics_params,
                                  int n_dynamics,

                                  ldouble * priors,

                                  ldouble * total_log_likelihood,
                                  ldouble * side_no_peak_likelihoods) {

    int status = SUCCESS;

    ldouble ** F = matrix(n_timepoints, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: F"); exit(1); }

    ldouble ** B = matrix(n_timepoints, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: B"); exit(1); }

    ldouble ** emission_cache = matrix(n_timepoints, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for matrix: emission_cache"); exit(1); }

    calculate_signal_cache_split(block_fgr,
                                 block_covariates,
                                 block_length,
                                 n_timepoints,
                                 n_covariates,
                                 foreground_delta,
                                 foreground_betas,
                                 background_delta,
                                 background_betas,
                                 emission_cache);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward_split(emission_cache,
                               n_timepoints,
                               block_length,
                               dynamics_params,
                               priors,
                               F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward_split(emission_cache,
                                    n_timepoints,
                                    block_length,
                                    dynamics_params,
                                    priors,
                                    B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }

    *total_log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        *total_log_likelihood = add_log_probs(*total_log_likelihood, F[n_timepoints - 1][i]);
    }

    *(side_no_peak_likelihoods + n_timepoints - 1) = F[n_timepoints - 1][0];

    for (int t = 0; t < n_timepoints - 1; t ++ ) {

        int cur_end = 0;

        ldouble total_log_prob = -INF;

        for (int next_end = 0; next_end < block_length + 1; next_end ++ ) {

            int end_dist = next_end - cur_end;

            ldouble log_prob = F[t][cur_end] +
                               boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                               B[t + 1][next_end] +
                               emission_cache[t + 1][next_end];

            total_log_prob = add_log_probs(total_log_prob, log_prob);

        }

        *(side_no_peak_likelihoods + t) = total_log_prob;

    }

    CLEANUP:

    free_matrix(emission_cache, n_timepoints);
    free_matrix(F, n_timepoints);
    free_matrix(B, n_timepoints);

    return status;

}


int get_no_peak_likelihoods (
             int * block_fgr,
             ldouble * block_covariates,
             int block_length,

             int n_covariates,
             int n_timepoints,

             ldouble * foreground_delta,
             ldouble * foreground_betas,

             ldouble * background_delta,
             ldouble * background_betas,

             ldouble * dynamics_params,
             int n_dynamics,

             ldouble * priors,

             ldouble * total_log_likelihood,
             ldouble * no_peak_likelihoods) {

    int status = SUCCESS;

    // calculate the forward and backward matrices for each cluster
    ldouble *** F = cube(n_timepoints, block_length + 1, block_length + 1);
    if (F == NULL) { fprintf(stderr, "Cannot allocate memory for cube: F"); exit(1); }

    ldouble *** B = cube(n_timepoints, block_length + 1, block_length + 1);
    if (B == NULL) { fprintf(stderr, "Cannot allocate memory for cube: B"); exit(1); }

    ldouble *** emission_cache = cube(n_timepoints, block_length + 1, block_length + 1);
    if (emission_cache == NULL) { fprintf(stderr, "Cannot allocate memory for cube: emission_cache"); exit(1); }

    for (int t = 0; t < n_timepoints; t ++) {
        *(no_peak_likelihoods + t) = -INF;
    }

    status = calculate_signal_cache(block_fgr,
                                    block_covariates,
                                    block_length,
                                    n_timepoints,
                                    n_covariates,
                                    foreground_delta,
                                    foreground_betas,
                                    background_delta,
                                    background_betas,
                                    emission_cache);

    int fwd_status = -1, bck_status = -1;

    fwd_status = forward(emission_cache,
                         n_timepoints,
                         block_length,
                         dynamics_params,
                         priors,
                         F);

    if (fwd_status == SUCCESS) {
        // calculate the backward matrix
        bck_status = backward(emission_cache,
                              n_timepoints,
                              block_length,
                              dynamics_params,
                              priors,
                              B);
    }

    if (fwd_status != SUCCESS || bck_status != SUCCESS) {
        status = E_UNDERFLOW;
        goto CLEANUP;
    }

    *total_log_likelihood = -INF;
    for (int i = 0; i < block_length + 1; i ++ ) {
        for (int j = i; j < block_length + 1; j ++ ) {
            *total_log_likelihood = add_log_probs(*total_log_likelihood, F[n_timepoints - 1][i][j]);
        }
        *(no_peak_likelihoods + n_timepoints - 1) = add_log_probs(*(no_peak_likelihoods + n_timepoints - 1),
                                                                  F[n_timepoints - 1][i][i]);
    }

    for (int t = 0; t < n_timepoints - 1; t ++ ) {

        for (int cur_start = 0; cur_start < block_length + 1; cur_start ++ ) {

            for (int cur_end = cur_start; cur_end < block_length + 1; cur_end ++ ) {

                for (int next_start = 0; next_start < block_length + 1; next_start ++ ) {

                    int start_dist = cur_start - next_start;

                    for (int next_end = next_start; next_end < block_length + 1; next_end ++ ) {

                        int end_dist = next_end - cur_end;

                        ldouble log_prob = F[t][cur_start][cur_end] +
                                   boundary_movement_model(start_dist, t, dynamics_params, priors, n_timepoints) +
                                   boundary_movement_model(end_dist, t, dynamics_params, priors, n_timepoints) +
                                   B[t + 1][next_start][next_end] +
                                   emission_cache[t + 1][next_start][next_end];

                        if (cur_start == cur_end) {
                            *(no_peak_likelihoods + t) = add_log_probs(*(no_peak_likelihoods + t), log_prob);
                        }
                    }
                }

            }
        }
    }

    CLEANUP:

    free_cube(F, n_timepoints, block_length + 1);
    free_cube(B, n_timepoints, block_length + 1);
    free_cube(emission_cache, n_timepoints, block_length + 1);

    return status;
}



