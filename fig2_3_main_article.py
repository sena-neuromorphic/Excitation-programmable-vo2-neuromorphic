import numpy as np
import matplotlib.pyplot as plt

# -------------------- Proximity Function --------------------
def ProFuncVO2(x, gama):
    return 0.5*(1 - np.sin(gama*x))*(1 + np.tanh(np.pi**2 - 2*np.pi*x))

# -------------------- Automatic Anchor (major loop) --------------------
def anchor_major_auto(T0, T1, w, Tc, beta, gama, Tpr0=None, eps_g=1e-6):
    """
    Builds (Tr, gr, Tpr, delta) at the initial point t0 to start
    directly on the MAJOR LOOP.

    Strategy:
    Selects Tpr0 (typical major-loop scale, default Tpr0 = w)
    and computes gr using the inverse model equation at the
    reversal point T = Tr = T0:

        gr = 0.5 + 0.5 * tanh(
                beta * (
                    delta*w/2 + Tc - (Tr + Tpr0*P(0))
                )
             )

    Then the evolution starts using the standard rule,
    where Tpr is recalculated at each reversal point.
    """

    # Initial direction (heating/cooling)
    dT = T1 - T0
    delta = +1 if dT > 0 else -1 if dT < 0 else +1

    Tr = float(T0)

    if Tpr0 is None:
        Tpr0 = float(w)  # Typical choice for the major loop

    P0 = ProFuncVO2(0.0, gama)  # ~1

    arg0 = delta*w/2 + Tc - (Tr + Tpr0*P0)

    gr = 0.5 + 0.5*np.tanh(beta*arg0)
    gr = float(np.clip(gr, eps_g, 1.0 - eps_g))

    # Initial Tpr value
    Tpr = float(Tpr0)

    return Tr, gr, Tpr, delta

# -------------------- Hysteresis Update --------------------
def update_hysteresis_given_T(T, w, Tc, beta, gama,
                              Tpr0=None, eps_dT=1e-9):

    N = len(T)

    g = np.zeros(N, dtype=float)
    delta_vec = np.zeros(N, dtype=int)
    Tr_vec = np.zeros(N, dtype=float)
    Tpr_vec = np.zeros(N, dtype=float)

    # Automatic anchor: starts inside the major loop
    Tr, gr, Tpr, delta = anchor_major_auto(
        T[0], T[1], w, Tc, beta, gama, Tpr0=Tpr0
    )

    g[0] = gr
    delta_vec[0] = delta
    Tr_vec[0] = Tr
    Tpr_vec[0] = Tpr

    for n in range(1, N):

        dT = T[n] - T[n-1]

        delta_new = (
            +1 if dT > eps_dT
            else (-1 if dT < -eps_dT else delta)
        )

        # If reversal occurs:
        # define new reversal point (Tr, gr)
        # and recompute Tpr using the standard rule
        if delta_new != delta:

            Tr = T[n-1]

            gr = float(np.clip(g[n-1], 1e-6, 1.0 - 1e-6))

            delta = delta_new

            P0 = ProFuncVO2(0.0, gama)

            Tpr = (
                delta*w/2 + Tc - Tr
                - np.arctanh(2*gr - 1)/beta
            ) / max(P0, 1e-9)

            if abs(Tpr) < 1e-9:
                Tpr = np.sign(Tpr)*1e-9 if Tpr != 0 else 1e-3

        # Update g at point n
        Tpr_safe = np.sign(Tpr) * max(abs(Tpr), 1e-9)

        x = (T[n] - Tr) / Tpr_safe

        arg = (
            delta*w/2 + Tc
            - (T[n] + Tpr_safe*ProFuncVO2(x, gama))
        )

        g[n] = 0.5 + 0.5*np.tanh(beta*arg)

        g[n] = float(np.clip(g[n], 0.0, 1.0))

        delta_vec[n] = delta
        Tr_vec[n] = Tr
        Tpr_vec[n] = Tpr

    return g, delta_vec, Tr_vec, Tpr_vec

# ===================== VO2 Parameters =====================
w, Tc, beta, gama = 6.5, 47.6, 0.2, 0.9

Rs, Rm = 17.0, 140.0

# ===================== Temperature Excitation T(t) =====================
t = np.linspace(0, 5, 1000)

A, tau = 38.0, 0.35

T = 50.0 + A*np.exp(-tau*t)*np.sin(2*np.pi*t)

# ===================== Run Hysteresis =====================
# To make the loop wider/narrower,
# adjust Tpr0 (e.g., 0.7*w, w, 1.3*w)

g, delta_vec, Tr_vec, Tpr_vec = update_hysteresis_given_T(
    T, w, Tc, beta, gama, Tpr0=w
)

# ===================== R(T) =====================
R = g*Rs*np.exp(2553.0/(T + 273.0)) + Rm

R = np.maximum(R, 1e-9)

# ===================== Plots =====================

# Temperature excitation
plt.figure()

plt.plot(t, T)

plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')

plt.title('Temperature Excitation')

plt.grid(True)

plt.show()

# Hysteresis loop
plt.figure()

plt.plot(T, g)

plt.xlabel('Temperature (°C)')
plt.ylabel('Volume Fraction g')

plt.title('Hysteresis g × T (VO₂) - automatic start at Major-Loop')

plt.grid(True)

plt.show()

# Resistance as function of temperature

plt.figure()

plt.plot(T, R/1e3)  # Convert Ω to kΩ

plt.xlabel('Temperature (°C)')
plt.ylabel('Resistance (kΩ)')

plt.title('R(T)')

plt.grid(True)

plt.show()

# Optional: branch inspection
plt.figure()

plt.step(t, delta_vec, where='post')

plt.ylim([-1.2, 1.2])

plt.xlabel('Time (s)')
plt.ylabel('Branch Indicator δ')

plt.title('Branch Evolution (+1 Heating, -1 Cooling)')

plt.grid(True)

plt.show()