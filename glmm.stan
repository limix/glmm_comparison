data {
    int<lower=0> N; // number of samples
    int<lower=0> P; // number of SNPs
    int nsuc[N];
    int ntri[N];
    matrix[N, P] G;
}

parameters {
    vector[N] e;
    vector[P] u;
    real offset;
    real<lower=0.0001> sigma_g;
    real<lower=0.0001> sigma_e;
}

model {
    vector[N] z;
    e ~ normal(0, sigma_e);
    u ~ normal(0, sigma_g);
    z = offset + G * u + e;
    nsuc ~ binomial_logit(ntri, z);
}
