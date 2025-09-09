import numpy as np
from scipy import optimize
from scipy.special import digamma, gamma
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------#

class StudentsTMixtureModel:
    def __init__(self, n_components=2, max_iter=100, tol=1e-8, 
                 init_method='kmeans', dof_lower_bound=1.0, dof_upper_bound=100.0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.dof_lower_bound = dof_lower_bound
        self.dof_upper_bound = dof_upper_bound
        
        self.weights = None
        self.means = None
        self.variances = None
        self.dofs = None
        self.responsibilities = None
        
    def _student_t_pdf(self, x, mu, sigma2, nu):
        """Exact Student's-t probability density function."""
        # Handle very small variances
        sigma2 = max(sigma2, 1e-10)
        
        # Normalization constant
        const = (gamma((nu + 1) / 2) / 
                (np.sqrt(nu * np.pi * sigma2) * gamma(nu / 2)))
        
        # PDF value
        pdf_val = const * (1 + (x - mu)**2 / (nu * sigma2)) ** (-(nu + 1) / 2)
        
        return pdf_val
    
    def _initialize_parameters(self, X):
        """Parameter initialization."""
        n_samples = len(X)
        
        if self.init_method == 'kmeans' and n_samples >= self.n_components:
            kmeans = KMeans(n_clusters=self.n_components, n_init=10)
            labels = kmeans.fit_predict(X.reshape(-1, 1))
            
            self.weights = np.zeros(self.n_components)
            self.means = np.zeros(self.n_components)
            self.variances = np.zeros(self.n_components)
            self.dofs = np.ones(self.n_components) * 10.0
            
            for k in range(self.n_components):
                cluster_data = X[labels == k]
                self.weights[k] = len(cluster_data) / n_samples
                self.means[k] = np.mean(cluster_data) if len(cluster_data) > 0 else np.mean(X)
                self.variances[k] = np.var(cluster_data) if len(cluster_data) > 1 else np.var(X)
                
        else:
            # Fallback initialization!
            self.weights = np.ones(self.n_components) / self.n_components
            self.means = np.linspace(np.min(X), np.max(X), self.n_components)
            self.variances = np.ones(self.n_components) * np.var(X) / self.n_components
            self.dofs = np.ones(self.n_components) * 10.0
    
    def _update_dofs(self, X, responsibilities, u_values):
        """Update of degrees of freedom using root finding."""
        new_dofs = np.copy(self.dofs)
        
        for k in range(self.n_components):
            r_k = responsibilities[:, k]
            u_k = u_values[:, k]
            n_k = np.sum(r_k)
            
            if n_k < 1e-10:  # Avoiding division by zero.
                new_dofs[k] = self.dofs[k]
                continue
            
            # Defining the equation to solve for degrees of freedom.
            def dof_equation(nu):
                term1 = np.log(nu/2) - digamma(nu/2) + 1
                term2 = (np.sum(r_k * (np.log(u_k) - u_k))) / n_k
                term3 = digamma((self.dofs[k] + 1)/2) - np.log((self.dofs[k] + 1)/2)
                return term1 + term2 + term3
            
            # Using Brent's method for root finding.
            try:
                solution = optimize.root_scalar(
                    dof_equation,
                    bracket=[self.dof_lower_bound, self.dof_upper_bound],
                    method='brentq'
                )
                new_dofs[k] = solution.root
            except (ValueError, RuntimeError):
                # Fallback: use previous value if root finding fails!
                new_dofs[k] = self.dofs[k]
        
        return new_dofs
    
    def fit(self, X):
        """Completing EM algorithm for Student's-t mixture model."""
        n_samples = len(X)
        
        # Initialize the parameters.
        self._initialize_parameters(X)
        
        # EM algorithm.
        prev_log_likelihood = -np.inf
        log_likelihoods = []
        
        for _ in range(self.max_iter):
            # E-step: Compute responsibilities and u values.
            responsibilities = np.zeros((n_samples, self.n_components))
            u_values = np.zeros((n_samples, self.n_components))
            
            # Compute component densities.
            component_densities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                component_densities[:, k] = self._student_t_pdf(
                    X, self.means[k], self.variances[k], self.dofs[k]
                )
            
            # Compute responsibilities.
            weighted_densities = self.weights * component_densities
            responsibilities = weighted_densities / np.sum(weighted_densities, axis=1, keepdims=True)
            
            # Handle numerical issues.
            responsibilities = np.nan_to_num(responsibilities, nan=1.0/self.n_components)
            
            # Compute u values (variance factors).
            for k in range(self.n_components):
                # Avoiding division by 0.
                variance = max(self.variances[k], 1e-8)

                u_values[:, k] = (self.dofs[k] + 1) / (
                    self.dofs[k] + (X - self.means[k])**2 / variance
                )
            
            # M-step: Update parameters.
            n_k = responsibilities.sum(axis=0)
            new_weights = n_k / n_samples
            
            new_means = np.zeros(self.n_components)
            new_variances = np.zeros(self.n_components)
            
            for k in range(self.n_components):
                # Update mean.
                weighted_sum = np.sum(responsibilities[:, k] * u_values[:, k])
                if weighted_sum > 1e-10:
                    new_means[k] = np.sum(responsibilities[:, k] * u_values[:, k] * X) / weighted_sum
                else:
                    new_means[k] = self.means[k]
                
                # Update variance.
                if n_k[k] > 1e-10:
                    new_variances[k] = np.sum(
                        responsibilities[:, k] * u_values[:, k] * (X - new_means[k])**2
                    ) / n_k[k]
                else:
                    new_variances[k] = self.variances[k]
            
            # Updating the degrees of freedom.
            new_dofs = self._update_dofs(X, responsibilities, u_values)
            
            # Constraining degrees of freedom.
            new_dofs = np.clip(new_dofs, self.dof_lower_bound, self.dof_upper_bound)
            
            # Computing log-likelihood for convergence check.
            log_likelihood = 0
            for k in range(self.n_components):
                log_likelihood += np.sum(
                    responsibilities[:, k] * np.log(
                        self.weights[k] * self._student_t_pdf(
                            X, self.means[k], self.variances[k], self.dofs[k]
                        ) + 1e-300
                    )
                )
            
            log_likelihoods.append(log_likelihood)
            
            # Checking convergence.
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = log_likelihood
            
            # Updating parameters.
            self.weights = new_weights
            self.means = new_means
            self.variances = new_variances
            self.dofs = new_dofs
            self.responsibilities = responsibilities
        
        return self, log_likelihoods
    
    def predict_proba(self, X):
        """Predicting class probabilities."""
        n_samples = len(X)
        probabilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            probabilities[:, k] = self.weights[k] * self._student_t_pdf(
                X, self.means[k], self.variances[k], self.dofs[k]
            )
        
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        """Predicting class membership."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def get_threshold(self):
        """
        Finding the optimal threshold 
        between the two distributions.
        """
        if self.n_components != 2:
            raise ValueError("Threshold finding only implemented for 2 components")
        
        # Finding the intersection point of the two distributions.
        def threshold_equation(x):
            pdf1 = self.weights[0] * self._student_t_pdf(x, self.means[0], self.variances[0], self.dofs[0])
            pdf2 = self.weights[1] * self._student_t_pdf(x, self.means[1], self.variances[1], self.dofs[1])
            return pdf1 - pdf2
        
        # Searching for the root between the two means.
        lower_bound = min(self.means)
        upper_bound = max(self.means)
        
        try:
            solution = optimize.root_scalar(
                threshold_equation,
                bracket=[lower_bound, upper_bound],
                method='brentq'
            )
            return solution.root
        except (ValueError, RuntimeError):
            # Fallback: weighted average of means!
            return np.average(self.means, weights=self.weights)
