model = """

data {

    int<lower=0> n_a;           // observations a
    int<lower=0> n_b;           // observations b
    
    vector[n_a] a;              // a vector
    vector[n_b] b;              // b vector
    
    real mu;                    // mean prior parameter
    real s;                     // standard deviation prior parameter
    real phi;                   // nu prior parameter
    
    real alpha;                 // sigma prior parameter
    real beta;                  // sigma prior parameter 
    
    }
    
parameters {

    real mu_a;                  // mu_a parameter
    real mu_b;                  // mu_b parameter 
    
    real<lower=0> sigma_a;      // sigma a parameter
    real<lower=0> sigma_b;      // sigma b parameter 
    
    real<lower=1> nu;           // nu parameter
    
    }
    
model {

    // Priors
    
    mu_a ~ normal(mu, 2 * s);
    mu_b ~ normal(mu, 2 * s);
    
    sigma_a ~ inv_gamma(alpha, beta);
    sigma_b ~ inv_gamma(alpha, beta);
    
    // exponential + 1 distribution
    target += log(phi) - ((nu - 1) * phi); 
     
    // Likelihood
    
    a ~ student_t(nu, mu_a, sigma_a);
    b ~ student_t(nu, mu_b, sigma_b);

    }

generated quantities {

    real mu_delta;              // mu_b - mu_a
    mu_delta = mu_b - mu_a;

    }

"""
