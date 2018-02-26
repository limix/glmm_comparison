data {
    int<lower=0> N; // number of samples
    int nsuc[N];
    int ntri[N];
    matrix[N, N] K;
    vector[N] g;
}
transformed data {
    matrix[N, N] L;
    L = cholesky_decompose(K);
}
parameters {
    vector[N] e;
    vector[N] u_effsiz;
    real offset;
    real snp_effect;
    real<lower=0.0001> sigma_g;
    real<lower=0.0001> sigma_e;
}
transformed parameters {
    vector[N] u;
    u  = L * u_effsiz;
}
model {
    vector[N] z;

    e ~ normal(0, sigma_e);
    u_effsiz ~ normal(0, sigma_g);

    z = offset + g * snp_effect + u + e;
    nsuc ~ binomial_logit(ntri, z);
}
