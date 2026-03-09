"""
Synthetic Data Generation for PPLS Model
========================================

This module generates synthetic data following the PPLS model structure with sine function-based 
loading matrices. Uses a simplified directory structure with direct output to the specified
data directory without complex nested folders.

Architecture Overview:
---------------------
The module provides the SineDataGenerator class for generating PPLS data with:
- Sine function-based loading matrices W and C
- Gaussian noise following the standard PPLS model
- Direct data persistence to specified output directory

Function List:
--------------
SineDataGenerator:
    - __init__(p, q, r, n_samples, random_seed, output_dir): Initialize data generation
    - generate_sine_loadings(frequency, magnitude): Create W and C matrices using sine functions
    - generate_true_parameters(sigma_e2, sigma_f2, sigma_h2): Generate all true PPLS parameters
    - generate_samples(params): Generate X, Y samples from PPLS model
    - _generate_latent_variables(params): Generate latent variables T, U, H
    - _apply_noise(X_clean, Y_clean, params): Add Gaussian noise to observations
    - save_true_parameters(params): Save ground truth parameters
    - _orthonormalize_loadings(W, C): Ensure orthonormal columns

Call Relationships:
------------------
SineDataGenerator.generate_samples() → PPLSModel.sample()
SineDataGenerator.generate_true_parameters() → SineDataGenerator.generate_sine_loadings()
SineDataGenerator.generate_sine_loadings() → SineDataGenerator._orthonormalize_loadings()
"""

import numpy as np
from scipy.linalg import orth
from typing import Dict, Tuple, Optional
import json
import os
import pickle

from .ppls_model import PPLSModel


class SineDataGenerator:
    """
    Generate PPLS data with sine function-based loading matrices.
    Implements the simulation setup with simplified output structure.
    """
    
    def __init__(self, p: int = 20, q: int = 20, r: int = 3, 
                 n_samples: int = 500, random_seed: int = 42,
                 output_dir: str = "./data"):
        """
        Initialize data generator.
        
        Parameters:
        -----------
        p : int
            Dimension of x (default: 20)
        q : int
            Dimension of y (default: 20)
        r : int
            Number of latent variables (default: 3)
        n_samples : int
            Number of samples to generate (default: 500)
        random_seed : int
            Random seed for reproducibility
        output_dir : str
            Directory for saving data (default: "./data")
        """
        self.p = p
        self.q = q
        self.r = r
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.output_dir = output_dir
        
        # Model instance
        self.model = PPLSModel(p, q, r)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_sine_loadings(self, frequency: float = 0.7, 
                              magnitude: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate W and C matrices using sine functions.
        
        According to the paper: "The columns of both W and C are based on a sine function,
        with slight distortions due to the orthonormalization by the Gram-Schmidt procedure."
        
        Parameters:
        -----------
        frequency : float
            Frequency parameter for sine function
        magnitude : float
            Magnitude scaling factor
            
        Returns:
        --------
        W : np.ndarray of shape (p, r)
            Loading matrix for X
        C : np.ndarray of shape (q, r)
            Loading matrix for Y
        """
        # Generate base sine patterns for W
        W = np.zeros((self.p, self.r))
        for j in range(self.r):
            # Create sine wave with different phase for each component
            phase = j * np.pi / self.r
            x = np.linspace(0, 2 * np.pi * frequency, self.p)
            W[:, j] = magnitude * np.sin(x + phase)
            
            # Add small random perturbation
            W[:, j] += 0.1 * np.random.randn(self.p)
            
        # Generate base sine patterns for C
        C = np.zeros((self.q, self.r))
        for j in range(self.r):
            # Create sine wave with different phase
            phase = j * np.pi / self.r + np.pi / 4  # Offset from W
            x = np.linspace(0, 2 * np.pi * frequency, self.q)
            C[:, j] = magnitude * np.sin(x + phase)
            
            # Add small random perturbation
            C[:, j] += 0.1 * np.random.randn(self.q)
            
        # Orthonormalize columns using Gram-Schmidt
        W, C = self._orthonormalize_loadings(W, C)
            
        return W, C
    
    def generate_true_parameters(self, 
                               sigma_e2: float = 0.1,
                               sigma_f2: float = 0.1,
                               sigma_h2: float = 0.05) -> Dict:
        """
        Generate all true PPLS model parameters.
        
        Parameters:
        -----------
        sigma_e2 : float
            Noise variance for X
        sigma_f2 : float
            Noise variance for Y
        sigma_h2 : float
            Noise variance for latent connection
            
        Returns:
        --------
        params : dict
            Dictionary containing all true parameters
        """
        # Generate loading matrices using sine functions
        W, C = self.generate_sine_loadings()
        
        # Generate B matrix (diagonal, positive, decreasing)
        # Ensure identifiability: (θ²_ti * b_i) decreasing
        b_values = np.linspace(2.0, 0.5, self.r)  # Decreasing values
        B = np.diag(b_values)
        
        # Generate Sigma_t (diagonal covariance of t)
        # Make θ²_ti such that (θ²_ti * b_i) is decreasing
        theta_t2_values = np.linspace(1.0, 0.3, self.r)
        
        # Verify identifiability constraint
        products = theta_t2_values * b_values
        assert all(products[i] > products[i+1] for i in range(self.r-1)), \
            "Identifiability constraint violated: (θ²_ti * b_i) must be decreasing"
            
        Sigma_t = np.diag(theta_t2_values)
        
        # Store all parameters
        params = {
            'W': W,
            'C': C,
            'B': B,
            'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2,
            'sigma_f2': sigma_f2,
            'sigma_h2': sigma_h2,
            'p': self.p,
            'q': self.q,
            'r': self.r,
            'n_samples': self.n_samples
        }
        
        return params
    
    def generate_samples(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate X, Y samples from PPLS model.
        
        Parameters:
        -----------
        params : dict
            Model parameters
            
        Returns:
        --------
        X : np.ndarray of shape (n_samples, p)
            Generated input data
        Y : np.ndarray of shape (n_samples, q)
            Generated output data
        """
        # Use PPLSModel to generate samples
        X, Y = self.model.sample(
            n_samples=self.n_samples,
            W=params['W'],
            C=params['C'],
            B=params['B'],
            Sigma_t=params['Sigma_t'],
            sigma_e2=params['sigma_e2'],
            sigma_f2=params['sigma_f2'],
            sigma_h2=params['sigma_h2']
        )
        
        return X, Y
    
    def _generate_latent_variables(self, params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate latent variables T, U, H.
        
        Parameters:
        -----------
        params : dict
            Model parameters
            
        Returns:
        --------
        T : np.ndarray of shape (n_samples, r)
            Latent variables for X
        U : np.ndarray of shape (n_samples, r)
            Latent variables for Y
        H : np.ndarray of shape (n_samples, r)
            Noise in latent space
        """
        # Generate T ~ N(0, Sigma_t)
        theta_t = np.sqrt(np.diag(params['Sigma_t']))
        T = np.random.randn(self.n_samples, self.r) @ np.diag(theta_t)
        
        # Generate H ~ N(0, sigma_h2 * I)
        H = np.sqrt(params['sigma_h2']) * np.random.randn(self.n_samples, self.r)
        
        # Compute U = T*B + H
        U = T @ params['B'] + H
        
        return T, U, H
    
    def _apply_noise(self, X_clean: np.ndarray, Y_clean: np.ndarray, 
                    params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply noise to clean observations.
        
        Parameters:
        -----------
        X_clean : np.ndarray
            Clean X data (T @ W.T)
        Y_clean : np.ndarray
            Clean Y data (U @ C.T)
        params : dict
            Model parameters including noise variances
            
        Returns:
        --------
        X : np.ndarray
            Noisy X data
        Y : np.ndarray
            Noisy Y data
        """
        # Add Gaussian noise
        E = np.sqrt(params['sigma_e2']) * np.random.randn(self.n_samples, self.p)
        F = np.sqrt(params['sigma_f2']) * np.random.randn(self.n_samples, self.q)
        
        X = X_clean + E
        Y = Y_clean + F
        
        return X, Y
    
    def save_true_parameters(self, params: Dict):
        """
        Save true parameters to output directory.
        
        Parameters:
        -----------
        params : dict
            True parameters
        """
        # Save as pickle for exact preservation
        with open(os.path.join(self.output_dir, 'ground_truth.pkl'), 'wb') as f:
            pickle.dump(params, f)
            
        # Save individual numpy arrays
        np.save(os.path.join(self.output_dir, 'ground_truth_W.npy'), params['W'])
        np.save(os.path.join(self.output_dir, 'ground_truth_C.npy'), params['C'])
        np.save(os.path.join(self.output_dir, 'ground_truth_B.npy'), params['B'])
        np.save(os.path.join(self.output_dir, 'ground_truth_Sigma_t.npy'), params['Sigma_t'])
        
        # Save as JSON for easy inspection (converting arrays to lists)
        params_serializable = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params_serializable[key] = value.tolist()
            else:
                params_serializable[key] = value
                
        with open(os.path.join(self.output_dir, 'ground_truth_parameters.json'), 'w') as f:
            json.dump(params_serializable, f, indent=4)
            
        # Save parameter summary
        summary = {
            "W_shape": list(params['W'].shape),
            "C_shape": list(params['C'].shape),
            "B_diagonal": np.diag(params['B']).tolist(),
            "Sigma_t_diagonal": np.diag(params['Sigma_t']).tolist(),
            "sigma_e2": params['sigma_e2'],
            "sigma_f2": params['sigma_f2'],
            "sigma_h2": params['sigma_h2'],
            "identifiability_products": (np.diag(params['Sigma_t']) * np.diag(params['B'])).tolist(),
            "random_seed": self.random_seed,
            "n_samples": self.n_samples
        }
        
        with open(os.path.join(self.output_dir, 'ground_truth_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    def _orthonormalize_loadings(self, W: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure W and C have orthonormal columns using Gram-Schmidt.
        
        Parameters:
        -----------
        W : np.ndarray of shape (p, r)
        C : np.ndarray of shape (q, r)
        
        Returns:
        --------
        W_orth, C_orth : orthonormalized matrices
        """
        W_orth = orth(W)
        C_orth = orth(C)
        
        # Handle case where orth returns fewer columns
        if W_orth.shape[1] < self.r:
            W_orth = W / np.linalg.norm(W, axis=0, keepdims=True)
        if C_orth.shape[1] < self.r:
            C_orth = C / np.linalg.norm(C, axis=0, keepdims=True)
            
        return W_orth, C_orth


class RandomOrthogonalDataGenerator:
    """Generate high-dimensional PPLS data with random orthogonal loadings."""

    def __init__(
        self,
        p: int,
        q: int,
        r: int,
        n_samples: int,
        random_seed: int = 42,
        output_dir: str = "./data",
    ):
        self.p = int(p)
        self.q = int(q)
        self.r = int(r)
        self.n_samples = int(n_samples)
        self.random_seed = int(random_seed)
        self.output_dir = output_dir
        self.model = PPLSModel(self.p, self.q, self.r)
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _random_orthonormal_matrix(n_rows: int, n_cols: int, rng: np.random.RandomState) -> np.ndarray:
        mat = rng.randn(n_rows, n_cols)
        q_mat, r_mat = np.linalg.qr(mat)
        diag = np.sign(np.diag(r_mat))
        diag[diag == 0] = 1.0
        return q_mat[:, :n_cols] * diag[:n_cols]

    @staticmethod
    def _default_theta_t2_values(r: int) -> np.ndarray:
        return 0.5 * np.linspace(float(r), 1.0, int(r))

    @staticmethod
    def _default_b_values(r: int) -> np.ndarray:
        return np.linspace(2.0, 0.5, int(r))

    def generate_true_parameters(
        self,
        sigma_e2: float = 0.1,
        sigma_f2: float = 0.1,
        sigma_h2: float = 0.05,
        theta_t2_values: Optional[np.ndarray] = None,
        b_values: Optional[np.ndarray] = None,
    ) -> Dict:
        rng = np.random.RandomState(self.random_seed)

        W = self._random_orthonormal_matrix(self.p, self.r, rng)
        C = self._random_orthonormal_matrix(self.q, self.r, rng)

        theta_t2_values = np.asarray(
            theta_t2_values if theta_t2_values is not None else self._default_theta_t2_values(self.r),
            dtype=float,
        )
        b_values = np.asarray(
            b_values if b_values is not None else self._default_b_values(self.r),
            dtype=float,
        )

        if theta_t2_values.shape != (self.r,):
            raise ValueError(f"theta_t2_values must have shape ({self.r},), got {theta_t2_values.shape}")
        if b_values.shape != (self.r,):
            raise ValueError(f"b_values must have shape ({self.r},), got {b_values.shape}")
        if np.any(theta_t2_values <= 0) or np.any(b_values <= 0):
            raise ValueError("theta_t2_values and b_values must be strictly positive")

        products = theta_t2_values * b_values
        if not np.all(products[:-1] > products[1:]):
            raise ValueError("Identifiability constraint violated: theta_t2_values * b_values must be strictly decreasing")

        return {
            'W': W,
            'C': C,
            'B': np.diag(b_values),
            'Sigma_t': np.diag(theta_t2_values),
            'sigma_e2': float(sigma_e2),
            'sigma_f2': float(sigma_f2),
            'sigma_h2': float(sigma_h2),
            'p': self.p,
            'q': self.q,
            'r': self.r,
            'n_samples': self.n_samples,
            'generator': 'random_orthogonal',
            'random_seed': self.random_seed,
            'identifiability_products': products,
        }

    def generate_samples(self, params: Dict, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_seed if seed is None else int(seed))
        theta_t = np.sqrt(np.diag(params['Sigma_t']))
        T = rng.randn(self.n_samples, self.r) @ np.diag(theta_t)
        H = np.sqrt(float(params['sigma_h2'])) * rng.randn(self.n_samples, self.r)
        U = T @ params['B'] + H
        E = np.sqrt(float(params['sigma_e2'])) * rng.randn(self.n_samples, self.p)
        F = np.sqrt(float(params['sigma_f2'])) * rng.randn(self.n_samples, self.q)
        X = T @ params['W'].T + E
        Y = U @ params['C'].T + F
        return X, Y

    def save_true_parameters(self, params: Dict):
        with open(os.path.join(self.output_dir, 'ground_truth.pkl'), 'wb') as f:
            pickle.dump(params, f)

        np.save(os.path.join(self.output_dir, 'ground_truth_W.npy'), params['W'])
        np.save(os.path.join(self.output_dir, 'ground_truth_C.npy'), params['C'])
        np.save(os.path.join(self.output_dir, 'ground_truth_B.npy'), params['B'])
        np.save(os.path.join(self.output_dir, 'ground_truth_Sigma_t.npy'), params['Sigma_t'])

        params_serializable = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params_serializable[key] = value.tolist()
            else:
                params_serializable[key] = value

        with open(os.path.join(self.output_dir, 'ground_truth_parameters.json'), 'w') as f:
            json.dump(params_serializable, f, indent=4)

        summary = {
            'W_shape': list(params['W'].shape),
            'C_shape': list(params['C'].shape),
            'B_diagonal': np.diag(params['B']).tolist(),
            'Sigma_t_diagonal': np.diag(params['Sigma_t']).tolist(),
            'sigma_e2': float(params['sigma_e2']),
            'sigma_f2': float(params['sigma_f2']),
            'sigma_h2': float(params['sigma_h2']),
            'identifiability_products': (np.diag(params['Sigma_t']) * np.diag(params['B'])).tolist(),
            'random_seed': int(self.random_seed),
            'n_samples': int(self.n_samples),
            'generator': 'random_orthogonal',
        }

        with open(os.path.join(self.output_dir, 'ground_truth_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)