
def _stan_model():

    return """
    
    data {

        int<lower=0> n_x;   // Observations X
        int<lower=0> n_y;   // Observations Y

        vector[n_x] X;      // X vector
        vector[n_y] Y;      // Y vector   

        real alpha;         // alpha mu prior parameter 
        real beta;          // beta mu prior parameter

        real delta;         // delta sigma prior parameter
        real gamma;         // gamma sigma prior parameter

        real epsilon;        // epsilon exponetial prior parameter

        }

    parameters {

        real mu_x;              // mu_x parameter
        real mu_y;              // mu_y parameter 
    
        real<lower=0> sigma_x;  // sigma x parameter
        real<lower=0> sigma_y;  // sigma y parameter 
    
        real<lower=1> nu;       // nu parameter

        }

    model {

        // Priors
    
        mu_x ~ normal(alpha, beta);
        mu_y ~ normal(alpha, beta);
    
        sigma_x ~ inv_gamma(delta, gamma);
        sigma_y ~ inv_gamma(delta, gamma);
    
        nu ~ exponential(epsilon);    

        // Likelihoods

        X ~ student_t(nu, mu_x, sigma_x);
        Y ~ student_t(nu, mu_y, sigma_y);

        }
        """
    