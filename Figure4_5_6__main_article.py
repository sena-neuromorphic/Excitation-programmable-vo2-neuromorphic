#Excitation programmable electrothermal hysteretic operating regimes in #VO₂ devices
#Memristive and Neuromorphic Devices

#B. A. S. F. Sena 1,2,+,* and L. A. L. de Almeida1,+

#1 Federal University of ABC (UFABC), Center for Engineering, Modeling and
#Applied Social Sciences, Santo Andre, SP, 09210-580, Brazil

#2 Federal Institute of Sao Paulo (IFSP), Department of Electrical,
#São Paulo, SP, 01109-010, Brazil ˜

#* sena.bruno@ifsp.edu.br
#+ these authors contributed equally to this work


#Reproduces Figures 4, 5, and 6 from the main article. The script also #generates additional figures currently under evaluation for inclusion #either in the main manuscript, the Supplementary Information, or future #versions of the work.

#This script was originally executed in a Google Colab environment;
#running it there is recommended for faster and smoother execution.

"""
================================================================================
Excitation programmable electrothermal hysteretic operating regimes in VO₂ devices (article)
================================================================================

This code implements the coupled electro-thermal circuit simulation for VO2
memristors using the Limiting Loop Proximity (LLP) hysteresis model.

Two canonical excitation modes are implemented:
  - Force-V (Thevenin drive): Voltage source with series resistance
  - Force-I (Norton drive): Current source injection


Author: Bruno Aparecido Sousa Figueiredo Sena and Luiz Alberto Luz de Almeida
Date: 01/2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Callable, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MATERIAL AND CIRCUIT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLPParams:
    """
    LLP-VO2 Hysteresis Model Parameters.

    These parameters define the major loop shape and the proximity-based
    internal trajectory behavior.
    """
    # Hysteresis shape parameters
    w: float = 6.5              # Hysteresis width [K or °C]
    Tc: float = 320.75          # Critical temperature [K] (≈47.6°C)
    beta: float = 0.25          # Slope parameter [K^-1]
    gamma: float = 0.99         # Proximity kernel parameter

    # EMA (Effective Medium Approximation) parameters
    R0: float = 17.0            # Semiconducting base resistance [Ω]
    R_m: float = 140.0          # Metallic phase resistance [Ω]
    E_a: float = 0.22           # Activation energy [eV]

    # Derived constants
    k_B: float = 8.617e-5       # Boltzmann constant [eV/K]

    @property
    def E_a_over_kB(self) -> float:
        """Activation energy ratio for Arrhenius law."""
        return self.E_a / self.k_B  # ≈ 2553 K


@dataclass
class CircuitParams:
    """
    Electro-Thermal Circuit Parameters.

    These define the lumped electrical and thermal network elements
    shown in the paper's circuit topology.
    """
    # Thermal network
    C_th: float = 1.0e-5        # Thermal capacitance [J/K]
    G_th: float = 4e-4          # Thermal conductance [W/K]
    T_sub: float = 318.15       # Substrate temperature [K] (45°C)

    # Electrical network
    R_s: float = 500.0          # Series resistance [Ω]
    R_parallel: float = 1e6    # Leakage/shunt resistance [Ω]
    C_parallel: float = 1e-10   # Parasitic capacitance [F]

    @property
    def tau_th(self) -> float:
        """Thermal time constant [s]."""
        return self.C_th / self.G_th


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LLP PROXIMITY FUNCTION AND HYSTERESIS OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def proximity_kernel(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Proximity function P(x) for the LLP model.

    This kernel controls how fast internal trajectories relax toward
    the limiting (major) loop after a reversal.

    P(x) = (1/2)(1 - sin(γx))(1 + tanh(π² - 2πx))

    Properties:
      - P(0) > 0 (finite proximity at reversal)
      - P(x) → 0 as |x| → ∞ (asymptotic collapse to major loop)
      - Monotone decreasing for positive x

    Parameters
    ----------
    x : array_like
        Reduced coordinate (T - T_r) / T_pr
    gamma : float
        Proximity decay rate parameter

    Returns
    -------
    P : array_like
        Proximity function values
    """
    return 0.5 * (1.0 - np.sin(gamma * x)) * (1.0 + np.tanh(np.pi**2 - 2.0*np.pi*x))


def compute_P0(gamma: float) -> float:
    """Compute P(0) for normalization in the anchoring condition."""
    return proximity_kernel(0.0, gamma)


def conductance_fraction(T: np.ndarray, T_r: float, T_pr: float,
                         delta: int, params: LLPParams) -> np.ndarray:
    """
    Compute the semiconducting volume fraction g(T) using LLP.

    This is the core hysteresis mapping that determines the internal
    trajectory based on the current LLP state (δ, T_r, T_pr).

    g(T) = 0.5 + 0.5·tanh(β·(δ·w/2 + T_c - T_eff))

    where T_eff = T + T_pr·P((T - T_r)/T_pr)

    Parameters
    ----------
    T : float or array
        Current temperature [K]
    T_r : float
        Reversal temperature (left anchor) [K]
    T_pr : float
        Proximity temperature scale [K]
    delta : int
        Branch direction (+1 = heating, -1 = cooling)
    params : LLPParams
        Model parameters

    Returns
    -------
    g : float or array
        Semiconducting phase fraction [0, 1]
    """
    # Compute reduced coordinate with safe division
    T_pr_safe = np.sign(T_pr) * max(abs(T_pr), 1e-9)
    x = (T - T_r) / T_pr_safe

    # Effective temperature including proximity shift
    P_x = proximity_kernel(x, params.gamma)
    T_eff = T + T_pr * P_x

    # Major loop mapping with branch offset
    arg = params.beta * (delta * params.w / 2.0 + params.Tc - T_eff)
    g = 0.5 + 0.5 * np.tanh(arg)

    return np.clip(g, 0.0, 1.0)


def resistance_from_g(T: np.ndarray, g: np.ndarray, params: LLPParams) -> np.ndarray:
    """
    Compute effective resistance R(T, g) using the EMA model.

    R(T) = R_m + g(T)·R_s(T)

    where R_s(T) = R_0·exp(E_a / (k_B·T))  [Arrhenius law]

    This high-contrast effective medium approximation captures the
    metallic (R_m) and semiconducting (temperature-activated) channels.

    Parameters
    ----------
    T : float or array
        Temperature [K]
    g : float or array
        Semiconducting phase fraction
    params : LLPParams
        Model parameters

    Returns
    -------
    R : float or array
        Effective resistance [Ω]
    """
    # Arrhenius law for semiconducting channel resistance
    R_s = params.R0 * np.exp(params.E_a_over_kB / T)

    # EMA composite resistance
    R = params.R_m + g * R_s

    return R


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ELECTRO-THERMAL ODE SYSTEMS (FORCE-V AND FORCE-I)
# ═══════════════════════════════════════════════════════════════════════════════

class LLPState:
    """
    Encapsulates the LLP hysteresis operator state.

    The state consists of:
      - delta: Current branch direction (+1 heating, -1 cooling)
      - T_r: Temperature at last reversal (left anchor)
      - g_r: Conductance fraction at last reversal
      - T_pr: Proximity temperature scale

    This state is updated only at genuine thermal reversals.
    """

    def __init__(self, params: LLPParams, T_init: float, g_init: float = 0.5):
        """
        Initialize LLP state at a given temperature.

        Parameters
        ----------
        params : LLPParams
            Model parameters
        T_init : float
            Initial temperature [K]
        g_init : float
            Initial conductance fraction (default: 0.5)
        """
        self.params = params
        self.P0 = compute_P0(params.gamma)

        # Initial state assumes heating direction
        self.delta = +1
        self.T_r = T_init
        self.g_r = g_init

        # Compute initial T_pr from anchoring condition
        self.T_pr = self._compute_Tpr(self.delta, self.T_r, self.g_r)

    def _compute_Tpr(self, delta: int, T_r: float, g_r: float) -> float:
        """
        Compute proximity temperature from the anchoring condition.

        T_pr = [δ·w/2 + T_c - T_r - β^{-1}·arctanh(2g_r - 1)] / P(0)

        This ensures the internal trajectory passes through (T_r, g_r).
        """
        # Safe clipping to avoid arctanh singularities
        g_r_safe = np.clip(g_r, 1e-9, 1.0 - 1e-9)

        numerator = (delta * self.params.w / 2.0
                    + self.params.Tc
                    - T_r
                    - (1.0/self.params.beta) * np.arctanh(2.0*g_r_safe - 1.0))

        T_pr = numerator / max(self.P0, 1e-9)

        # Signed clipping for numerical stability
        return np.sign(T_pr) * max(abs(T_pr), 1e-9)

    def clone(self) -> 'LLPState':
        """Create a snapshot of the current state for clone-and-evaluate."""
        new_state = LLPState.__new__(LLPState)
        new_state.params = self.params
        new_state.P0 = self.P0
        new_state.delta = self.delta
        new_state.T_r = self.T_r
        new_state.g_r = self.g_r
        new_state.T_pr = self.T_pr
        return new_state

    def update_if_reversal(self, T_prev: float, T_curr: float,
                          g_prev: float, eps_dT: float = 1e-6) -> bool:
        """
        Check for thermal reversal and update state if detected.

        A genuine reversal occurs when the sign of dT/dt changes with
        sufficient magnitude (above the deadband eps_dT).

        Hard left-anchoring: The new anchor is (T_{n-1}, g_{n-1}).

        Parameters
        ----------
        T_prev : float
            Temperature at previous step
        T_curr : float
            Temperature at current step
        g_prev : float
            Conductance at previous step
        eps_dT : float
            Deadband threshold to suppress noise-induced toggling

        Returns
        -------
        reversed : bool
            True if a reversal was detected and state updated
        """
        dT = T_curr - T_prev

        # Determine new direction with deadband
        if dT > eps_dT:
            delta_new = +1
        elif dT < -eps_dT:
            delta_new = -1
        else:
            delta_new = self.delta  # No change

        # Check for genuine reversal
        if delta_new != self.delta:
            # Hard left-anchoring at (T_prev, g_prev)
            self.T_r = T_prev
            self.g_r = np.clip(g_prev, 1e-9, 1.0 - 1e-9)
            self.delta = delta_new
            self.T_pr = self._compute_Tpr(self.delta, self.T_r, self.g_r)
            return True

        return False

    def compute_g(self, T: float) -> float:
        """Compute g(T) using current LLP state."""
        return conductance_fraction(T, self.T_r, self.T_pr, self.delta, self.params)

    def compute_R(self, T: float, g: Optional[float] = None) -> float:
        """Compute R(T) using current LLP state."""
        if g is None:
            g = self.compute_g(T)
        return resistance_from_g(T, g, self.params)


def rhs_force_v(t: float, y: np.ndarray,
                v_in_func: Callable[[float], float],
                llp_frozen: LLPState,
                llp_params: LLPParams,
                circuit: CircuitParams) -> np.ndarray:
    """
    Right-hand side for Force-V (Thevenin) mode.

    State vector: y = [v_e, T]

    Governing ODEs:

      dv_e/dt = (1/C_∥) · [(v_in - v_e)/R_s - v_e/R_∥ - v_e/R(T)]

      dT/dt = (1/C_th) · [v_e²/R(T) - G_th·(T - T_sub)]

    The LLP state is frozen during integration (clone-and-evaluate).

    Parameters
    ----------
    t : float
        Current time [s]
    y : array
        State vector [v_e, T]
    v_in_func : callable
        Input voltage function v_in(t)
    llp_frozen : LLPState
        Frozen snapshot of LLP state
    llp_params : LLPParams
        LLP model parameters
    circuit : CircuitParams
        Circuit parameters

    Returns
    -------
    dydt : array
        Time derivatives [dv_e/dt, dT/dt]
    """
    v_e, T = y

    # Compute R(T) using frozen LLP state
    g = llp_frozen.compute_g(T)
    R_T = llp_frozen.compute_R(T, g)

    # Input voltage at current time
    v_in = v_in_func(t)

    # Electrical node equation (Force-V: Thevenin)
    # Current through series resistor: (v_in - v_e) / R_s
    # Current through shunt resistor: v_e / R_∥
    # Current through VO2 channel: v_e / R(T)
    i_Rs = (v_in - v_e) / circuit.R_s
    i_Rp = v_e / circuit.R_parallel
    i_VO2 = v_e / R_T

    dv_e_dt = (i_Rs - i_Rp - i_VO2) / circuit.C_parallel

    # Thermal equation
    # Joule heating: P_J = v_e² / R(T)
    P_joule = v_e**2 / R_T
    P_loss = circuit.G_th * (T - circuit.T_sub)

    dT_dt = (P_joule - P_loss) / circuit.C_th

    return np.array([dv_e_dt, dT_dt])


def rhs_force_i(t: float, y: np.ndarray,
                i_in_func: Callable[[float], float],
                llp_frozen: LLPState,
                llp_params: LLPParams,
                circuit: CircuitParams) -> np.ndarray:
    """
    Right-hand side for Force-I (Norton) mode.

    State vector: y = [v_e, T]

    Governing ODEs:

      dv_e/dt = (1/C_∥) · [i_in - v_e·(1/R_s + 1/R_∥ + 1/R(T))]

      dT/dt = (1/C_th) · [v_e²/R(T) - G_th·(T - T_sub)]

    Parameters
    ----------
    t : float
        Current time [s]
    y : array
        State vector [v_e, T]
    i_in_func : callable
        Input current function i_in(t)
    llp_frozen : LLPState
        Frozen snapshot of LLP state
    llp_params : LLPParams
        LLP model parameters
    circuit : CircuitParams
        Circuit parameters

    Returns
    -------
    dydt : array
        Time derivatives [dv_e/dt, dT/dt]
    """
    v_e, T = y

    # Compute R(T) using frozen LLP state
    g = llp_frozen.compute_g(T)
    R_T = llp_frozen.compute_R(T, g)

    # Input current at current time
    i_in = i_in_func(t)

    # Electrical node equation (Force-I: Norton)
    # Total conductance seen by current source
    G_total = 1.0/circuit.R_s + 1.0/circuit.R_parallel + 1.0/R_T

    dv_e_dt = (i_in - v_e * G_total) / circuit.C_parallel

    # Thermal equation (same as Force-V)
    P_joule = v_e**2 / R_T
    P_loss = circuit.G_th * (T - circuit.T_sub)

    dT_dt = (P_joule - P_loss) / circuit.C_th

    return np.array([dv_e_dt, dT_dt])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EVENT-DRIVEN SIMULATION WITH CLONE-AND-EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_electrothermal(t_span: Tuple[float, float],
                            t_eval: np.ndarray,
                            excitation_func: Callable[[float], float],
                            mode: str,
                            llp_params: LLPParams,
                            circuit: CircuitParams,
                            y0: Optional[np.ndarray] = None,
                            method: str = 'BDF') -> dict:
    """
    Simulate the coupled electro-thermal LLP-VO2 system.

    This function implements the event-driven, clone-and-evaluate
    architecture. The LLP state is frozen
    during each macro-step and updated only at step boundaries
    when genuine thermal reversals are detected.

    Parameters
    ----------
    t_span : tuple
        Integration interval (t_start, t_end)
    t_eval : array
        Times at which to store the solution
    excitation_func : callable
        Input function: v_in(t) for Force-V, i_in(t) for Force-I
    mode : str
        'force_v' or 'force_i'
    llp_params : LLPParams
        LLP model parameters
    circuit : CircuitParams
        Circuit parameters
    y0 : array, optional
        Initial state [v_e0, T0]. Default: [0, T_sub]
    method : str
        ODE solver method ('BDF', 'Radau', 'RK45')

    Returns
    -------
    results : dict
        Dictionary containing:
        - t: time array
        - v_e: device voltage
        - T: temperature
        - g: conductance fraction
        - R: resistance
        - i_e: device current
        - P_joule: dissipated power
        - delta: branch direction
        - reversals: list of reversal times
    """
    # Initialize
    if y0 is None:
        y0 = np.array([0.0, circuit.T_sub])

    # Create LLP state
    llp_state = LLPState(llp_params, y0[1], g_init=0.5)

    # Storage arrays
    n_points = len(t_eval)
    results = {
        't': t_eval.copy(),
        'v_e': np.zeros(n_points),
        'T': np.zeros(n_points),
        'g': np.zeros(n_points),
        'R': np.zeros(n_points),
        'i_e': np.zeros(n_points),
        'P_joule': np.zeros(n_points),
        'delta': np.zeros(n_points, dtype=int),
        'reversals': []
    }

    # Initial values
    results['v_e'][0] = y0[0]
    results['T'][0] = y0[1]
    results['g'][0] = llp_state.compute_g(y0[1])
    results['R'][0] = llp_state.compute_R(y0[1], results['g'][0])
    results['i_e'][0] = y0[0] / results['R'][0] if results['R'][0] > 0 else 0
    results['P_joule'][0] = y0[0]**2 / results['R'][0] if results['R'][0] > 0 else 0
    results['delta'][0] = llp_state.delta

    # Select RHS function based on mode
    if mode.lower() == 'force_v':
        rhs_base = rhs_force_v
    elif mode.lower() == 'force_i':
        rhs_base = rhs_force_i
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'force_v' or 'force_i'")

    # Macro-step integration
    y_current = y0.copy()

    for n in range(1, n_points):
        t0 = t_eval[n-1]
        t1 = t_eval[n]

        # 1) Freeze LLP state for this macro-step (clone-and-evaluate)
        llp_frozen = llp_state.clone()

        # 2) Define RHS with frozen LLP state
        def rhs(t, y):
            return rhs_base(t, y, excitation_func, llp_frozen, llp_params, circuit)

        # 3) Integrate over macro-step
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y_current,
            method=method,
            t_eval=[t1],
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            print(f"Warning: Integration failed at t={t0:.4f}")
            # Use simple Euler fallback
            dt = t1 - t0
            dy = rhs(t0, y_current)
            y_new = y_current + dt * dy
        else:
            y_new = sol.y[:, -1]

        # Extract state
        v_e_new, T_new = y_new

        # 4) Event-driven LLP update at step boundary
        g_prev = results['g'][n-1]
        T_prev = results['T'][n-1]

        reversed = llp_state.update_if_reversal(T_prev, T_new, g_prev)
        if reversed:
            results['reversals'].append(t1)

        # 5) Compute outputs using updated LLP state
        g_new = llp_state.compute_g(T_new)
        R_new = llp_state.compute_R(T_new, g_new)
        i_new = v_e_new / R_new if R_new > 0 else 0
        P_new = v_e_new**2 / R_new if R_new > 0 else 0

        # Store results
        results['v_e'][n] = v_e_new
        results['T'][n] = T_new
        results['g'][n] = g_new
        results['R'][n] = R_new
        results['i_e'][n] = i_new
        results['P_joule'][n] = P_new
        results['delta'][n] = llp_state.delta

        # Update current state
        y_current = y_new

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXCITATION WAVEFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def sinusoidal_voltage(t: float, V_amp: float = 8.0, freq: float = 0.5,
                       V_offset: float = 0.0) -> float:
    """Sinusoidal voltage source for Force-V mode."""
    return V_offset + V_amp * np.sin(2.0 * np.pi * freq * t)


def damped_sinusoidal_voltage(t: float, V_amp: float = 8.0, freq: float = 1.0,
                              decay: float = 0.3) -> float:
    """Damped sinusoidal voltage (as in original code)."""
    return V_amp * np.exp(-decay * t) * np.sin(2.0 * np.pi * freq * t)


def triangular_voltage(t: float, V_amp: float = 8.0, period: float = 2.0) -> float:
    """Triangular wave voltage source for clean hysteresis loops."""
    phase = (t % period) / period
    if phase < 0.5:
        return V_amp * (4.0 * phase - 1.0)  # Rising
    else:
        return V_amp * (3.0 - 4.0 * phase)  # Falling


def sinusoidal_current(t: float, I_amp: float = 5e-3, freq: float = 0.5,
                       I_offset: float = 0.0) -> float:
    """Sinusoidal current source for Force-I mode."""
    return I_offset + I_amp * np.sin(2.0 * np.pi * freq * t)


def triangular_current(t: float, I_amp: float = 5e-3, period: float = 2.0) -> float:
    """Triangular wave current source for Force-I mode."""
    phase = (t % period) / period
    if phase < 0.5:
        return I_amp * (4.0 * phase - 1.0)
    else:
        return I_amp * (3.0 - 4.0 * phase)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN SIMULATION AND VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main simulation routine comparing Force-V and Force-I modes.

    This demonstrates the key differences in hysteresis behavior

    """
    print("="*75)
    print("ELECTRO-THERMAL LLP-VO2 SIMULATION: FORCE-V vs FORCE-I")
    print("="*75)

    # ─────────────────────────────────────────────────────────────────────────
    # Setup parameters
    # ─────────────────────────────────────────────────────────────────────────

    # LLP model parameters (from paper)
    llp_params = LLPParams(
        w=6.7,           # Hysteresis width [K]
        Tc=320.75,       # Critical temperature [K]
        beta=0.25,       # Slope parameter
        gamma=0.99,      # Proximity kernel parameter
        R0=17.0,         # Semiconducting base resistance [Ω]
        R_m=1140.0,      # Metallic + R_E resistance [Ω]
        E_a=0.22         # Activation energy [eV]
    )

    # Circuit parameters for Force-V (low series resistance → voltage forcing)
    circuit_force_v = CircuitParams(
        C_th=1.0e-5,         # Thermal capacitance [J/K]
        G_th=4e-4,           # Thermal conductance [W/K]
        T_sub=318.15,       # Substrate at 45°C
        R_s=50.0,            # Series resistance: LOW for Force-V
        R_parallel=1e7,      # High leakage resistance
        C_parallel=1e-10     # Small parasitic capacitance
    )

    # Circuit parameters for Force-I (high series resistance → current forcing)
    circuit_force_i = CircuitParams(
        C_th=1.0e-5,
        G_th=4e-4,
        T_sub=318.15,
        R_s=10000.0,         # Series resistance: HIGH for Force-I
        R_parallel=1e7,
        C_parallel=1e-10
    )

    # Time parameters
    t_end = 4.0  # seconds
    n_points = 4000
    t_eval = np.linspace(0, t_end, n_points)

    # Excitation parameters
    V_amp = 8     # Voltage amplitude for Force-V [V]
    I_amp = 6e-3    # Current amplitude for Force-I [A]
    freq = 0.5      # Excitation frequency [Hz]

    # ─────────────────────────────────────────────────────────────────────────
    # Run Force-V simulation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/2] Running Force-V (Thevenin) simulation...")
    print(f"      R_s = {circuit_force_v.R_s} Ω (low → voltage forcing)")
    print(f"      V_amp = {V_amp} V, freq = {freq} Hz")

    def v_in_func(t):
        return sinusoidal_voltage(t, V_amp=V_amp, freq=freq)

    results_v = simulate_electrothermal(
        t_span=(0, t_end),
        t_eval=t_eval,
        excitation_func=v_in_func,
        mode='force_v',
        llp_params=llp_params,
        circuit=circuit_force_v,
        method='BDF'
    )

    print(f"      Detected {len(results_v['reversals'])} thermal reversals")
    print(f"      T range: [{results_v['T'].min():.2f}, {results_v['T'].max():.2f}] K")
    print(f"      R range: [{results_v['R'].min():.1f}, {results_v['R'].max():.1f}] Ω")

    # ─────────────────────────────────────────────────────────────────────────
    # Run Force-I simulation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/2] Running Force-I (Norton) simulation...")
    print(f"      R_s = {circuit_force_i.R_s} Ω (high → current forcing)")
    print(f"      I_amp = {I_amp*1000:.1f} mA, freq = {freq} Hz")

    def i_in_func(t):
        return sinusoidal_current(t, I_amp=I_amp, freq=freq)

    results_i = simulate_electrothermal(
        t_span=(0, t_end),
        t_eval=t_eval,
        excitation_func=i_in_func,
        mode='force_i',
        llp_params=llp_params,
        circuit=circuit_force_i,
        method='BDF'
    )

    print(f"      Detected {len(results_i['reversals'])} thermal reversals")
    print(f"      T range: [{results_i['T'].min():.2f}, {results_i['T'].max():.2f}] K")
    print(f"      R range: [{results_i['R'].min():.1f}, {results_i['R'].max():.1f}] Ω")

    # ─────────────────────────────────────────────────────────────────────────
    # Generate comprehensive comparison plots
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*75)
    print("Generating comparison plots...")
    print("-"*75)

    # Plot style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Color scheme
    color_v = '#2E86AB'   # Blue for Force-V
    color_i = '#E63946'   # Red for Force-I
    color_aux = '#4CAF50' # Green for auxiliary

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 1: Main Comparison (Force-V vs Force-I)
    # ═══════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(16, 12))
    gs1 = GridSpec(3, 3, figure=fig1, hspace=0.35, wspace=0.3)



    # Plot 1: Temperature vs Time
    ax1 = fig1.add_subplot(gs1[0, :2])
    ax1.plot(results_v['t'], results_v['T'] - 273.15, color=color_v,
             linewidth=2, label='Force-V (Thevenin)', alpha=0.9)
    ax1.plot(results_i['t'], results_i['T'] - 273.15, color=color_i,
             linewidth=2, linestyle='--', label='Force-I (Norton)', alpha=0.9)
    ax1.axhline(y=llp_params.Tc - 273.15, color='gray', linestyle=':',
                alpha=0.5, label=f'$T_c$ = {llp_params.Tc-273.15:.1f}°C')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.set_title('Temperature Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Device Voltage vs Time
    ax2 = fig1.add_subplot(gs1[0, 2])
    ax2.plot(results_v['t'], results_v['v_e'], color=color_v,
             linewidth=1.5, label='Force-V', alpha=0.8)
    ax2.plot(results_i['t'], results_i['v_e'], color=color_i,
             linewidth=1.5, linestyle='--', label='Force-I', alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Device Voltage $v_e$ (V)', fontsize=11)
    ax2.set_title('Voltage at VO2 Terminal', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: I-V Characteristic (Critical comparison!)
    ax3 = fig1.add_subplot(gs1[1, 0])
    i_v_mA = results_v['i_e'] * 1000  # Convert to mA
    i_i_mA = results_i['i_e'] * 1000
    ax3.plot(results_v['v_e'], i_v_mA, color=color_v,
             linewidth=2, label='Force-V', alpha=0.8)
    ax3.plot(results_i['v_e'], i_i_mA, color=color_i,
             linewidth=2, linestyle='--', label='Force-I', alpha=0.8)
    ax3.set_xlabel('Device Voltage $v_e$ (V)', fontsize=11)
    ax3.set_ylabel('Device Current $i_e$ (mA)', fontsize=11)
    ax3.set_title('I-V Characteristic\n(Hysteresis Loop)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: R-T Hysteresis
    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.semilogy(results_v['T'] - 273.15, results_v['R'], color=color_v,
                 linewidth=2, label='Force-V', alpha=0.8)
    ax4.semilogy(results_i['T'] - 273.15, results_i['R'], color=color_i,
                 linewidth=2, linestyle='--', label='Force-I', alpha=0.8)
    ax4.axvline(x=llp_params.Tc - 273.15, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Temperature (°C)', fontsize=11)
    ax4.set_ylabel('Resistance R (Ω)', fontsize=11)
    ax4.set_title('R-T Hysteresis', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: g-T Hysteresis
    ax5 = fig1.add_subplot(gs1[1, 2])
    ax5.plot(results_v['T'] - 273.15, results_v['g'], color=color_v,
             linewidth=2, label='Force-V', alpha=0.8)
    ax5.plot(results_i['T'] - 273.15, results_i['g'], color=color_i,
             linewidth=2, linestyle='--', label='Force-I', alpha=0.8)
    ax5.axvline(x=llp_params.Tc - 273.15, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Temperature (°C)', fontsize=11)
    ax5.set_ylabel('Semiconducting Fraction g', fontsize=11)
    ax5.set_title('g-T Characteristic', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Dissipated Power
    ax6 = fig1.add_subplot(gs1[2, 0])
    ax6.plot(results_v['t'], results_v['P_joule'] * 1000, color=color_v,
             linewidth=1.5, label='Force-V', alpha=0.8)
    ax6.plot(results_i['t'], results_i['P_joule'] * 1000, color=color_i,
             linewidth=1.5, linestyle='--', label='Force-I', alpha=0.8)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Joule Power (mW)', fontsize=11)
    ax6.set_title('Dissipated Power', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Resistance vs Time
    ax7 = fig1.add_subplot(gs1[2, 1])
    ax7.semilogy(results_v['t'], results_v['R'], color=color_v,
                 linewidth=1.5, label='Force-V', alpha=0.8)
    ax7.semilogy(results_i['t'], results_i['R'], color=color_i,
                 linewidth=1.5, linestyle='--', label='Force-I', alpha=0.8)
    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_ylabel('Resistance R (Ω)', fontsize=11)
    ax7.set_title('Resistance Evolution', fontsize=12)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # Plot 8: Branch Direction (delta)
    ax8 = fig1.add_subplot(gs1[2, 2])
    ax8.step(results_v['t'], results_v['delta'], color=color_v,
             linewidth=1.5, where='post', label='Force-V', alpha=0.8)
    ax8.step(results_i['t'], results_i['delta'], color=color_i,
             linewidth=1.5, where='post', linestyle='--', label='Force-I', alpha=0.8)
    ax8.set_xlabel('Time (s)', fontsize=11)
    ax8.set_ylabel('Branch Direction δ', fontsize=11)
    ax8.set_title('Hysteresis Branch', fontsize=12)
    ax8.set_yticks([-1, 0, 1])
    ax8.set_yticklabels(['Cooling (-1)', '0', 'Heating (+1)'])
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    for ax, lab in zip(axes_list, panel_labels):
        ax.text(0.01, 0.98, lab, transform=ax.transAxes,
               fontsize=13, fontweight='bold',
               va='top', ha='left')


    fig1.suptitle('FORCE-V vs FORCE-I: Electro-Thermal VO2 Memristor Comparison\n'
                  '(LLP Hysteresis Model with Clone-and-Evaluate)',
                  fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('vo2_force_v_vs_force_i_main.png', dpi=300, bbox_inches='tight')
    print("  → Saved: vo2_force_v_vs_force_i_main.png")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 2: Detailed I-V Analysis (Key Result)
    # ═══════════════════════════════════════════════════════════════════════
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = ['a', 'b', 'c', 'd']
    for ax, lab in zip(axes.flat, labels):
        ax.text(0.02, 0.95, lab, transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                va='top', ha='left')

    # Force-V I-V with annotations
    ax = axes[0, 0]
    ax.plot(results_v['v_e'], i_v_mA, color=color_v, linewidth=2.5, alpha=0.9)
    ax.scatter([results_v['v_e'][0]], [i_v_mA[0]], color='green', s=100,
               zorder=5, marker='o', label='Start')
    ax.set_xlabel('$v_e$ (V)', fontsize=12)
    ax.set_ylabel('$i_e$ (mA)', fontsize=12)
    ax.set_title(f'Force-V (Thevenin): $R_s$ = {circuit_force_v.R_s} Ω\n'
                 f'Wide Hysteresis Loop', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force-I I-V with annotations
    ax = axes[0, 1]
    ax.plot(results_i['v_e'], i_i_mA, color=color_i, linewidth=2.5, alpha=0.9)
    ax.scatter([results_i['v_e'][0]], [i_i_mA[0]], color='green', s=100,
               zorder=5, marker='o', label='Start')
    ax.set_xlabel('$v_e$ (V)', fontsize=12)
    ax.set_ylabel('$i_e$ (mA)', fontsize=12)
    ax.set_title(f'Force-I (Norton): $R_s$ = {circuit_force_i.R_s} Ω\n'
                 f'Narrow Loop, S-NDR Visible', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined I-V
    ax = axes[1, 0]
    ax.plot(results_v['v_e'], i_v_mA, color=color_v, linewidth=2,
            label=f'Force-V ($R_s$={circuit_force_v.R_s}Ω)', alpha=0.8)
    ax.plot(results_i['v_e'], i_i_mA, color=color_i, linewidth=2,
            label=f'Force-I ($R_s$={circuit_force_i.R_s}Ω)', alpha=0.8)
    ax.set_xlabel('$v_e$ (V)', fontsize=12)
    ax.set_ylabel('$i_e$ (mA)', fontsize=12)
    ax.set_title('Overlay: Effect of Load Line on Hysteresis',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Phase portrait: dT/dt vs T
    ax = axes[1, 1]
    dt = t_eval[1] - t_eval[0]
    dTdt_v = np.gradient(results_v['T'], dt)
    dTdt_i = np.gradient(results_i['T'], dt)
    ax.plot(results_v['T'] - 273.15, dTdt_v, color=color_v,
            linewidth=1.5, label='Force-V', alpha=0.7)
    ax.plot(results_i['T'] - 273.15, dTdt_i, color=color_i,
            linewidth=1.5, label='Force-I', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=llp_params.Tc - 273.15, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('dT/dt (K/s)', fontsize=12)
    ax.set_title('Phase Portrait: Thermal Dynamics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig2.suptitle('Detailed I-V Analysis: Force-V vs Force-I',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('vo2_iv_analysis.png', dpi=300, bbox_inches='tight')
    print("  → Saved: vo2_iv_analysis.png")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 3: Force-V Detailed View
    # ═══════════════════════════════════════════════════════════════════════
    fig3, axes = plt.subplots(2, 3, figsize=(15, 9))

    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for ax, lab in zip(axes.flat, labels):
        ax.text(0.02, 0.95, lab, transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                va='top', ha='left')

    # Input voltage
    v_in_array = np.array([v_in_func(t) for t in t_eval])
    axes[0, 0].plot(t_eval, v_in_array, 'k-', linewidth=2, label='$v_{in}$')
    axes[0, 0].plot(t_eval, results_v['v_e'], color=color_v, linewidth=1.5,
                    label='$v_e$ (device)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_title('Force-V: Input vs Device Voltage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Temperature
    axes[0, 1].plot(t_eval, results_v['T'] - 273.15, color=color_v, linewidth=2)
    axes[0, 1].axhline(y=llp_params.Tc - 273.15, color='r', linestyle='--',
                       alpha=0.5, label='$T_c$')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Force-V: Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Resistance log scale
    axes[0, 2].semilogy(t_eval, results_v['R'], color=color_v, linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Resistance R (Ω)')
    axes[0, 2].set_title('Force-V: Resistance (log)')
    axes[0, 2].grid(True, alpha=0.3)

    # Current
    axes[1, 0].plot(t_eval, i_v_mA, color=color_v, linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Current (mA)')
    axes[1, 0].set_title('Force-V: Device Current')
    axes[1, 0].grid(True, alpha=0.3)

    # Conductance fraction
    axes[1, 1].plot(t_eval, results_v['g'], color=color_v, linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('g (semiconducting fraction)')
    axes[1, 1].set_title('Force-V: Phase Fraction')
    axes[1, 1].grid(True, alpha=0.3)

    # I-V curve
    axes[1, 2].plot(results_v['v_e'], i_v_mA, color=color_v, linewidth=2)
    axes[1, 2].set_xlabel('$v_e$ (V)')
    axes[1, 2].set_ylabel('$i_e$ (mA)')
    axes[1, 2].set_title('Force-V: I-V Hysteresis')
    axes[1, 2].grid(True, alpha=0.3)

    fig3.suptitle(f'FORCE-V (Thevenin) Detailed Results: $R_s$ = {circuit_force_v.R_s} Ω',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('vo2_force_v_detailed.png', dpi=300, bbox_inches='tight')
    print("  → Saved: vo2_force_v_detailed.png")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 4: Force-I Detailed View
    # ═══════════════════════════════════════════════════════════════════════
    fig4, axes = plt.subplots(2, 3, figsize=(15, 9))

    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for ax, lab in zip(axes.flat, labels):
        ax.text(0.02, 0.95, lab, transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                va='top', ha='left')


    # Input current
    i_in_array = np.array([i_in_func(t) for t in t_eval]) * 1000  # mA
    axes[0, 0].plot(t_eval, i_in_array, 'k-', linewidth=2, label='$i_{in}$')
    axes[0, 0].plot(t_eval, i_i_mA, color=color_i, linewidth=1.5,
                    label='$i_e$ (device)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Current (mA)')
    axes[0, 0].set_title('Force-I: Input vs Device Current')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Temperature
    axes[0, 1].plot(t_eval, results_i['T'] - 273.15, color=color_i, linewidth=2)
    axes[0, 1].axhline(y=llp_params.Tc - 273.15, color='r', linestyle='--',
                       alpha=0.5, label='$T_c$')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Force-I: Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Resistance log scale
    axes[0, 2].semilogy(t_eval, results_i['R'], color=color_i, linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Resistance R (Ω)')
    axes[0, 2].set_title('Force-I: Resistance (log)')
    axes[0, 2].grid(True, alpha=0.3)

    # Voltage
    axes[1, 0].plot(t_eval, results_i['v_e'], color=color_i, linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Voltage (V)')
    axes[1, 0].set_title('Force-I: Device Voltage')
    axes[1, 0].grid(True, alpha=0.3)

    # Conductance fraction
    axes[1, 1].plot(t_eval, results_i['g'], color=color_i, linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('g (semiconducting fraction)')
    axes[1, 1].set_title('Force-I: Phase Fraction')
    axes[1, 1].grid(True, alpha=0.3)

    # I-V curve
    axes[1, 2].plot(results_i['v_e'], i_i_mA, color=color_i, linewidth=2)
    axes[1, 2].set_xlabel('$v_e$ (V)')
    axes[1, 2].set_ylabel('$i_e$ (mA)')
    axes[1, 2].set_title('Force-I: I-V Hysteresis')
    axes[1, 2].grid(True, alpha=0.3)

    fig4.suptitle(f'FORCE-I (Norton) Detailed Results: $R_s$ = {circuit_force_i.R_s} Ω',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('vo2_force_i_detailed.png', dpi=300, bbox_inches='tight')
    print("  → Saved: vo2_force_i_detailed.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Print statistical summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("SIMULATION RESULTS SUMMARY")
    print("="*75)

    print("\n📊 FORCE-V (Thevenin Drive) Statistics:")
    print(f"   Temperature:  [{results_v['T'].min()-273.15:.2f}, "
          f"{results_v['T'].max()-273.15:.2f}] °C")
    print(f"   Resistance:   [{results_v['R'].min():.1f}, "
          f"{results_v['R'].max():.1f}] Ω")
    print(f"   Current:      [{results_v['i_e'].min()*1000:.2f}, "
          f"{results_v['i_e'].max()*1000:.2f}] mA")
    print(f"   Voltage:      [{results_v['v_e'].min():.2f}, "
          f"{results_v['v_e'].max():.2f}] V")
    print(f"   g range:      [{results_v['g'].min():.4f}, {results_v['g'].max():.4f}]")
    print(f"   Reversals:    {len(results_v['reversals'])}")

    print("\n📊 FORCE-I (Norton Drive) Statistics:")
    print(f"   Temperature:  [{results_i['T'].min()-273.15:.2f}, "
          f"{results_i['T'].max()-273.15:.2f}] °C")
    print(f"   Resistance:   [{results_i['R'].min():.1f}, "
          f"{results_i['R'].max():.1f}] Ω")
    print(f"   Current:      [{results_i['i_e'].min()*1000:.2f}, "
          f"{results_i['i_e'].max()*1000:.2f}] mA")
    print(f"   Voltage:      [{results_i['v_e'].min():.2f}, "
          f"{results_i['v_e'].max():.2f}] V")
    print(f"   g range:      [{results_i['g'].min():.4f}, {results_i['g'].max():.4f}]")
    print(f"   Reversals:    {len(results_i['reversals'])}")

    print("="*75)

    plt.show()

    return results_v, results_i


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ADDITIONAL ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parametric_study_Rs(Rs_values: list, mode: str = 'force_v'):
    """
    Study the effect of series resistance on hysteresis loop width.

    This demonstrates how varying R_s affects the load line intersection
    with the NDR region, producing loops of varying width.

    Parameters
    ----------
    Rs_values : list
        List of R_s values to simulate
    mode : str
        'force_v' or 'force_i'
    """
    print("\n" + "="*75)
    print(f"PARAMETRIC STUDY: Effect of R_s on {mode.upper()} Hysteresis")
    print("="*75)

    llp_params = LLPParams(
        w=6.7, Tc=320.75, beta=0.25, gamma=0.99,
        R0=17.0, R_m=1140.0, E_a=0.22
    )

    t_eval = np.linspace(0, 4.0, 2000)

    results_list = []

    for Rs in Rs_values:
        print(f"\n  Simulating R_s = {Rs} Ω...")

        circuit = CircuitParams(
            C_th=1.0e-5, G_th=4e-4, T_sub=318.15,
            R_s=Rs, R_parallel=1e6, C_parallel=1e-10
        )

        if mode == 'force_v':
            def exc_func(t):
                return sinusoidal_voltage(t, V_amp=8.0, freq=0.5)
        else:
            def exc_func(t):
                return sinusoidal_current(t, I_amp=6e-3, freq=0.5)

        results = simulate_electrothermal(
            t_span=(0, 4.0), t_eval=t_eval,
            excitation_func=exc_func, mode=mode,
            llp_params=llp_params, circuit=circuit,
            method='BDF'
        )
        results['Rs'] = Rs
        results_list.append(results)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(Rs_values)))

    for res, color in zip(results_list, cmap):
        Rs = res['Rs']
        label = f'$R_s$ = {Rs} Ω'

        axes[0].plot(res['v_e'], res['i_e']*1000, color=color,
                     linewidth=1.5, label=label)
        axes[1].plot(res['T']-273.15, res['g'], color=color,
                     linewidth=1.5, label=label)
        axes[2].semilogy(res['T']-273.15, res['R'], color=color,
                        linewidth=1.5, label=label)

    axes[0].set_xlabel('$v_e$ (V)')
    axes[0].set_ylabel('$i_e$ (mA)')
    axes[0].set_title('I-V Characteristic')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_ylabel('g')
    axes[1].set_title('g-T Hysteresis')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Temperature (°C)')
    axes[2].set_ylabel('R (Ω)')
    axes[2].set_title('R-T Hysteresis')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Effect of Series Resistance on {mode.upper()} Mode',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'vo2_Rs_study_{mode}.png', dpi=300, bbox_inches='tight')
    print(f"\n  → Saved: vo2_Rs_study_{mode}.png")
    plt.show()

    return results_list


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run main comparison
    results_v, results_i = main()

    # Optional: Run parametric study
    # Rs_values = [50, 100, 500, 1000, 5000, 10000]
    # parametric_study_Rs(Rs_values, mode='force_v')