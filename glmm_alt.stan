data {
    int<lower=0> N; // number of samples
    int nsuc[N];
    int ntri[N];
    matrix[N, N] K;
    vector[N] g;
}

parameters {
    vector[N] e;
    vector[N] u;
    real offset;
    real snp_effect;
    real<lower=0.0001> sigma_g;
    real<lower=0.0001> sigma_e;
}

model {
    vector[N] z;
    vector[N] zeros;
    for (n in 1:N) {
        e[n] ~ normal(0, sigma_e);
        zeros[n] = 0;
    }
    u ~ multi_normal(zeros, K);
    z = offset + g * snp_effect + sigma_g * u + e;
    nsuc ~ binomial_logit(ntri, z);
}
