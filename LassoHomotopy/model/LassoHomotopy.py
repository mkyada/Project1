import numpy as np
import matplotlib.pyplot as plt

class LassoHomotopyModel():
    def __init__(self):
        pass

    def fit(self, X, y, max_steps=1000, tol=1e-6):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).flatten()
        n, p = X.shape

        # Center X and y
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Normalize columns of X, avoid division by zero
        X_norm = np.linalg.norm(X_centered, axis=0)
        X_norm_safe = np.where(X_norm < tol, 1.0, X_norm)
        X_scaled = X_centered / X_norm_safe

        # Initialize
        beta = np.zeros(p)
        active_set = []
        path_lambdas = []
        path_betas = []

        # Initial correlations and lambda (maximum absolute correlation)
        corr = X_scaled.T.dot(y_centered)
        lambda_val = np.max(np.abs(corr))
        path_lambdas.append(lambda_val)
        path_betas.append(beta.copy())

        # If lambda is zero (e.g. constant features), return trivial solution.
        if lambda_val < tol:
            beta_final = beta / X_norm_safe
            intercept = y_mean
            return LassoHomotopyResults(beta_final, intercept, path_lambdas, path_betas)

        for step in range(max_steps):
            # Compute residual and correlations
            r = y_centered - X_scaled.dot(beta)
            corr = X_scaled.T.dot(r)

            # Update active set: add any variable with correlation close to current lambda
            new_indices = np.where(np.abs(corr) >= lambda_val - tol)[0]
            for idx in new_indices:
                if idx not in active_set:
                    active_set.append(idx)
            if len(active_set) == 0:
                break

            # Signs for variables in active set
            s = np.sign(corr[active_set])
            X_A = X_scaled[:, active_set]
            Gram = X_A.T.dot(X_A)
            try:
                invGram = np.linalg.inv(Gram)
            except np.linalg.LinAlgError:
                break  # if singular, exit loop

            # Compute equiangular direction for the active set
            one_vec = s  # same as sign vector
            A_val = 1.0 / np.sqrt(np.dot(one_vec, invGram.dot(one_vec)))
            w = A_val * invGram.dot(one_vec)  # weights for active set
            u = X_A.dot(w)  # equiangular direction

            # Compute candidate step sizes for variables not yet active.
            gamma_candidates = []
            for j in range(p):
                if j in active_set:
                    continue
                aj = X_scaled[:, j].dot(u)
                candidate1 = (lambda_val - corr[j]) / (A_val - aj) if abs(A_val - aj) > tol else np.inf
                candidate2 = (lambda_val + corr[j]) / (A_val + aj) if abs(A_val + aj) > tol else np.inf
                if candidate1 > tol:
                    gamma_candidates.append(candidate1)
                if candidate2 > tol:
                    gamma_candidates.append(candidate2)
            # Also consider candidates from the active set when a coefficient would cross zero.
            gamma_active = []
            for i, idx in enumerate(active_set):
                if abs(w[i]) > tol:
                    candidate = -beta[idx] / w[i]
                    if candidate > tol:
                        gamma_active.append(candidate)
            gamma_list = gamma_candidates + gamma_active
            if len(gamma_list) == 0:
                gamma = lambda_val / A_val
            else:
                gamma = min(gamma_list)

            # Update beta for variables in the active set.
            for i, idx in enumerate(active_set):
                beta[idx] += gamma * w[i]

            # Decrease lambda accordingly.
            lambda_val = lambda_val - gamma * A_val
            if lambda_val < 0:
                lambda_val = 0
            path_lambdas.append(lambda_val)
            path_betas.append(beta.copy())

            if lambda_val <= tol:
                break

        # Adjust coefficients back to original scale and compute intercept.
        beta_final = beta / X_norm_safe
        intercept = y_mean - np.dot(X_mean, beta_final)
        return LassoHomotopyResults(beta_final, intercept, path_lambdas, path_betas)


class LassoHomotopyResults():
    def __init__(self, coef, intercept, path_lambdas, path_betas):
        self.coef = coef
        self.intercept = intercept
        self.path_lambdas = path_lambdas
        self.path_betas = path_betas

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self.intercept + X.dot(self.coef)

    def plot_path(self):
        # Plot the evolution of each coefficient versus lambda.
        path_betas = np.array(self.path_betas)
        lambdas = self.path_lambdas
        plt.figure()
        for i in range(path_betas.shape[1]):
            plt.plot(lambdas, path_betas[:, i], label=f'coef_{i}')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient value')
        plt.title('Regularization Path')
        plt.legend()
        plt.gca().invert_xaxis()  # since lambda decreases over iterations
        plt.show()
