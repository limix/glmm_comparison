data {
    int<lower=0> N; // number of samples
    int nsuc[N];
    int ntri[N];
    matrix[N, N] K;
}

parameters {
    vector[N] e;
    vector[N] u;
    real offset;
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
    z = offset + sigma_g * u + e;
    nsuc ~ binomial_logit(ntri, z);
}
