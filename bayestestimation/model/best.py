model = """

data {

    int<lower=0> n_x;           // observations x
    int<lower=0> n_y;           // observations y
    
    vector[n_x] x;              // x vector
    vector[n_y] y;              // y vector
    
    real mu;                    // mean prior parameter
    real s;                     // standard deviation prior parameter
    real lambda;                // nu prior parameter
    
    real alpha;                 // sigma prior parameter
    real beta;                  // sigma prior parameter 
    
    }
    
parameters {

    real mu_x;                  // mu_x parameter
    real mu_y;                  // mu_y parameter 
    
    real<lower=0> sigma_x;      // sigma x parameter
    real<lower=0> sigma_y;      // sigma y parameter 
    
    real<lower=1> nu;           // nu parameter
    
    }
    
model {

    // Priors
    
    mu_x ~ normal(mu, 2 * s);
    mu_y ~ normal(mu, 2 * s);
    
    sigma_x ~ inv_gamma(alpha, beta);
    sigma_y ~ inv_gamma(alpha, beta);
    
    // exponential + 1 distribution
    target += log(lambda) - ((nu - 1) * lambda); 
     
    // Likelihood
    
    x ~ student_t(nu, mu_x, sigma_x);
    y ~ student_t(nu, mu_y, sigma_y);

    }

"""