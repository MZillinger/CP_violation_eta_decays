import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import dblquad
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PhysicsConstants:
    """Immutable data class for physical constants."""
    MPI: float = 0.13957039
    META: float = 0.547862
    METAP: float = 0.95778
    FPI: float = 0.09232
    L5: float = 1.66e-3
    L8: float = 1.08e-3
    LAMBDA12: float = -0.51
    THETA: float = -0.2958333082130388
    PHI: float = 0.6594833099114707
    GAMMA_ETAP_ETAPIPI: float = 0.0799e-3
    DELTA_GAMMA_ETAP_ETAPIPI: float = 2.7177380300536696e-6

class EtaPrimeDecayModel:
    """
    Model for CP Violation in Eta-Prime decays.
    Includes Fit logic, Hessian error estimation, and Reduced Chi2.
    """

    def __init__(self, omnes_file_path: str, constants: PhysicsConstants = PhysicsConstants()):
        self.c = constants
        self.omnes_re: Optional[interp1d] = None
        self.omnes_im: Optional[interp1d] = None
        self.integrals: dict = {}
        self._load_omnes_functions(omnes_file_path)

    def _load_omnes_functions(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Omnes data file not found: {path}")
        try:
            data = np.loadtxt(path)
            # data structure: [s/MPI^2, Real, Imag]
            data_mandelstam_s = data[:, 0] * self.c.MPI**2
            self.omnes_re = interp1d(data_mandelstam_s, data[:, 1], kind="cubic", fill_value="extrapolate")
            self.omnes_im = interp1d(data_mandelstam_s, data[:, 2], kind="cubic", fill_value="extrapolate")
            logger.info("Omnes functions loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Omnes functions: {e}")
            raise

    # --- Physics Helper Functions ---
    @staticmethod
    def kallen(a: float, b: float, c: float) -> float:
        return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c

    @staticmethod
    def beta(s: float, m: float) -> float:
        val = 1 - 4 * m**2 / s
        return np.sqrt(val) if isinstance(val, (float, int)) and val >= 0 else np.sqrt(val.clip(min=0))

    # --- Partial Wave Decomposition ---
    def mj0const(self) -> float:
        c = self.c
        term1 = 3 * np.sin(c.PHI) * np.cos(c.PHI) * c.MPI**2 / (3 * c.FPI**2)
        term2 = np.sqrt(2/3) * (c.MPI/c.FPI)**2 * c.LAMBDA12 * \
                (np.sin(c.THETA)*np.cos(c.THETA) - np.sin(c.PHI)*np.cos(c.PHI))
        term3 = 3 * np.sin(c.PHI) * np.cos(c.PHI) * \
                (-4*c.MPI**2*c.L5/(3*c.FPI**4)*(2*c.MPI**2 + c.META**2 + c.METAP**2) + 16*(c.MPI/c.FPI)**4*c.L8)
        return term1 + term2 + term3

    def mj0spart(self, s: float) -> float:
        c = self.c
        prefactor = 3 * np.sin(c.PHI) * np.cos(c.PHI) * 4 / (3 * c.FPI**4)
        k_val = self.kallen(s, c.META**2, c.METAP**2)
        b_val = self.beta(s, c.MPI)
        inner = (s**2 - 2*c.MPI**4 - c.META**4 - c.METAP**4 + 
                 2/3 * k_val * b_val**2 + 
                 2*c.MPI**2 * (c.METAP**2 - c.META**2)**2 / s + 
                 2 * (c.METAP**2 - c.MPI**2) * (c.META**2 - c.MPI**2))
        return prefactor * inner

    def mj2(self, s: float, z: float) -> float:
        c = self.c
        prefactor = 3 * np.sin(c.PHI) * np.cos(c.PHI) * 4 / (3 * c.FPI**4)
        k_val = self.kallen(s, c.META**2, c.METAP**2)
        b_val = self.beta(s, c.MPI)
        return prefactor * k_val * b_val * b_val * (3 * z**2 - 1) / 6

    # --- Pre-computation ---
    def precompute_integrals(self) -> None:
        logger.info("Starting pre-computation of integrals...")
        c = self.c
        lim_z_low, lim_z_high = -1.0, 1.0
        lim_s_low, lim_s_high = 4 * c.MPI**2, (c.METAP - c.META)**2
        
        def omega_sq(s): return self.omnes_re(s)**2 + self.omnes_im(s)**2
        def ps_factor(s): return np.sqrt(self.kallen(s, c.META**2, c.METAP**2)) * self.beta(s, c.MPI)

        integrands = {
            'mj0_L23quad_alphaconst': lambda s, z: self.mj0spart(s)**2 * omega_sq(s) * ps_factor(s),
            'mj0_L23quad_alphalinear': lambda s, z: self.mj0spart(s)**2 * s * omega_sq(s) * ps_factor(s),
            'mj0_L23quad_alphaquad': lambda s, z: self.mj0spart(s)**2 * s**2 * omega_sq(s) * ps_factor(s),
            'mj0_L23const_alphaconst': lambda s, z: omega_sq(s) * ps_factor(s),
            'mj0_L23const_alphalinear': lambda s, z: s * omega_sq(s) * ps_factor(s),
            'mj0_L23const_alphaquad': lambda s, z: s**2 * omega_sq(s) * ps_factor(s),
            'mj0_L23linear_alphaconst': lambda s, z: self.mj0spart(s) * omega_sq(s) * ps_factor(s),
            'mj0_L23linear_alphalinear': lambda s, z: self.mj0spart(s) * s * omega_sq(s) * ps_factor(s),
            'mj0_L23linear_alphaquad': lambda s, z: self.mj0spart(s) * s**2 * omega_sq(s) * ps_factor(s),
            'mj2_L23quad_alphaconst': lambda s, z: self.mj2(s, z)**2 * ps_factor(s)
        }

        for name, func in integrands.items():
            val, _ = dblquad(func, lim_z_low, lim_z_high, lim_s_low, lim_s_high, epsrel=1e-10)
            self.integrals[name] = val
        logger.info("Integration complete.")

    # --- Fit Functions ---
    def decayrate_fit_function(self, L23: float, alpha: float) -> float:
        if not self.integrals: raise RuntimeError("Integrals not computed.")
        i, c, mj0_c = self.integrals, self.c, self.mj0const()
        term = (L23**2 * i['mj0_L23quad_alphaconst'] + 2 * alpha * L23**2 * i['mj0_L23quad_alphalinear'] +
                alpha**2 * L23**2 * i['mj0_L23quad_alphaquad'] + mj0_c**2 * i['mj0_L23const_alphaconst'] +
                2 * alpha * mj0_c**2 * i['mj0_L23const_alphalinear'] + alpha**2 * mj0_c**2 * i['mj0_L23const_alphaquad'] +
                2 * L23 * mj0_c * i['mj0_L23linear_alphaconst'] + 4 * alpha * L23 * mj0_c * i['mj0_L23linear_alphalinear'] +
                2 * alpha**2 * L23 * mj0_c * i['mj0_L23linear_alphaquad'] + L23**2 * i['mj2_L23quad_alphaconst'])
        return (1 / (512 * np.pi**3 * c.METAP**3)) * term

    def dalitzplot_fit_function(self, x, y, A, L23, alpha):
        c = self.c
        Q = c.METAP - c.META - 2 * c.MPI
        s = (c.METAP - c.META)**2 - 2 * c.MPI * c.METAP * Q * (1 + y) / (c.META + 2 * c.MPI)
        k_val = self.kallen(s, c.META**2, c.METAP**2)
        b_val = self.beta(s, c.MPI)
        z = -2 * c.METAP * Q * x / (np.sqrt(3) * b_val * np.sqrt(k_val))
        
        mj0_part = self.mj0spart(s)
        mj0_c = self.mj0const()
        omega_mag_sq = self.omnes_re(s)**2 + self.omnes_im(s)**2
        mj2_val = self.mj2(s, z)
        
        term1 = (L23 * mj0_part + mj0_c)**2 * (1 + alpha * s)**2 * omega_mag_sq
        term2 = L23**2 * mj2_val**2
        term3 = 2 * (L23 * mj0_part + mj0_c) * (1 + alpha * s) * self.omnes_re(s) * L23 * mj2_val
        return A * (term1 + term2 + term3)

    def chisquared(self, params, data_x, data_y, data_w, data_ew):
        A, L23, alpha = params
        theory_w = self.dalitzplot_fit_function(data_x, data_y, A, L23, alpha)
        safe_ew_sq = np.where(data_ew > 0, data_ew**2, 1.0)
        term1 = np.sum((theory_w - data_w)**2 / safe_ew_sq)
        calc_gamma = self.decayrate_fit_function(L23, alpha)
        term2 = (calc_gamma - self.c.GAMMA_ETAP_ETAPIPI)**2 / (self.c.DELTA_GAMMA_ETAP_ETAPIPI**2)
        return term1 + term2

    # --- Hessian & Statistics ---
    def compute_hessian(self, params, data_x, data_y, data_w, data_ew, epsilon=1e-4):
        """Calculates the Hessian matrix of the Chi-squared function using finite differences."""
        n = len(params)
        hessian = np.zeros((n, n))
        params = np.array(params, dtype=float) 
        steps = [abs(p) * epsilon + 1e-8 for p in params]
        
        def func(p):
            return self.chisquared(p, data_x, data_y, data_w, data_ew)

        f_0 = func(params)
        for i in range(n):
            for j in range(i, n):
                p_plus_i = params.copy(); p_minus_i = params.copy()
                if i == j: 
                    p_plus_i[i] += steps[i]; p_minus_i[i] -= steps[i]
                    f_plus = func(p_plus_i); f_minus = func(p_minus_i)
                    hessian[i, i] = (f_plus - 2*f_0 + f_minus) / (steps[i]**2)
                else: 
                    p_pp = params.copy(); p_pm = params.copy(); p_mp = params.copy(); p_mm = params.copy()
                    p_pp[i] += steps[i]; p_pp[j] += steps[j]
                    p_pm[i] += steps[i]; p_pm[j] -= steps[j]
                    p_mp[i] -= steps[i]; p_mp[j] += steps[j]
                    p_mm[i] -= steps[i]; p_mm[j] -= steps[j]
                    term1 = func(p_pp) - func(p_pm); term2 = func(p_mp) - func(p_mm)
                    hessian[i, j] = (term1 - term2) / (4 * steps[i] * steps[j])
                    hessian[j, i] = hessian[i, j]
        return hessian

    def fit(self, data_file_path: str, initial_guess: List[float] = [0.002, 0.001, -3.0]):
        if not os.path.exists(data_file_path): raise FileNotFoundError(f"{data_file_path} not found")
        logger.info(f"Loading data from {data_file_path}...")
        data = np.loadtxt(data_file_path, skiprows=1)
        xc, yc, w, ew = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        
        def objective(p): return self.chisquared(p, xc, yc, w, ew)
        
        logger.info("Starting Nelder-Mead optimization...")
        result = minimize(objective, initial_guess, method="Nelder-Mead")
        
        if result.success:
            logger.info("Fit successful. Calculating statistics...")
            
            # 1. Calculate Reduced Chi2
            # N_dof = N_data_points + N_constraints - N_free_parameters
            n_data = len(xc)
            n_constraints = 1 # Gamma decay rate
            n_params = len(initial_guess)
            n_dof = n_data + n_constraints - n_params
            
            result.reduced_chi2 = result.fun / n_dof
            result.n_dof = n_dof

            # 2. Calculate Errors using Hessian
            try:
                hessian = self.compute_hessian(result.x, xc, yc, w, ew)
                covariance = np.linalg.inv(hessian) 
                errors = np.sqrt(np.diag(covariance)) 
                
                result.hessian = hessian
                result.covariance = covariance
                result.errors = errors
            except np.linalg.LinAlgError:
                logger.warning("Hessian inversion failed. Covariance matrix undefined.")
                result.errors = np.full_like(result.x, np.nan)

        return result

# --- Entry Point ---
if __name__ == "__main__":
    OMNES_FILE = "data/Omnesfunctions-d00-d10-central.dat"
    DATA_FILE = "data/BESIII-2018-etap-to-pipieta-dalc.txt"
    
    if os.path.exists(OMNES_FILE) and os.path.exists(DATA_FILE):
        try:
            model = EtaPrimeDecayModel(OMNES_FILE)
            model.precompute_integrals()
            res = model.fit(DATA_FILE)
            
            print("\n" + "="*30)
            print(" FINAL FIT RESULTS")
            print("="*30)
            
            # Print Standard Chi2 and Reduced Chi2
            print(f"Chi2 Total:      {res.fun:.2f}")
            if hasattr(res, 'reduced_chi2'):
                print(f"Degrees of Freedom: {res.n_dof}")
                print(f"Reduced Chi2:    {res.reduced_chi2:.4f}")
                
                # Simple interpretation
                if 0.8 < res.reduced_chi2 < 1.2:
                    print(">> Status: Excellent Fit (Chi2/Ndf ~ 1)")
                elif res.reduced_chi2 > 2.0:
                    print(">> Status: Poor Fit (Underfitting or Model Mismatch)")
                elif res.reduced_chi2 < 0.5:
                    print(">> Status: Overfitting (Errors might be overestimated)")
            
            print("-" * 30)
            
            # Print parameters with errors
            if hasattr(res, 'errors'):
                param_names = ['A', 'L23', 'alpha']
                for name, val, err in zip(param_names, res.x, res.errors):
                    print(f"{name:<6}: {val:.4e} +/- {err:.4e}")
            else:
                print(f"Parameters: {res.x}")
            
            if hasattr(res, 'covariance'):
                print("\nCorrelation Matrix:")
                d = np.diag(np.sqrt(np.diag(res.covariance)))
                d_inv = np.linalg.inv(d)
                corr = d_inv @ res.covariance @ d_inv
                print(np.array2string(corr, precision=3, floatmode='fixed'))

        except Exception as e:
            logging.error(f"Error: {e}")
