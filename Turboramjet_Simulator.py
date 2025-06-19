# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:45:28 2025

@author: Liam Allen 46988601

Turboramjet performance calculator
"""

#%% IMPORT LIBRARIES
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

arr = np.array

#%% CONSTANTS

# SIMULATION
elements = 500
considerMbypassburn = False

# GENERAL ASSUMPTIONS
cpc = 1004.                     # [J/kgK]   Specific heat at constant pressure pre-burner
cpt = 1239.                     # [J/kgK]   Specific heat at constant pressure post-burner
kc = 1.4                        # []        Ratio of specific heat capacity pre-burner: Gamma
kt = 1.3                        # []        Ratio of specific heat capacity post-burner: Gamma
q0 = 50e3                       # [Pa]      Constant dynamic pressure flight trajectory
T0 = 250.                       # [K]       Freestream temperature
A0 = 14.4                       # [m2]      Engine capture area. Note there are 2x engines
cd = 0.03                       # []        Vehicle drag coefficient
Aref = 382.                     # [m2]      Vehicle planform area
Rc = cpc * (1 - 1 / kc)         # [J/kgK]   Specific gas constant pre-burner
Rt = cpt * (1 - 1 / kt)         # [J/kgK]   Specific gas constant post-burner
a0 = math.sqrt(kc * Rc * T0)    # [m/s2]    Freestream speed of sound
P0P9 = 1.                       # []        Perfectly expanded core exhaust
P0P19 = 1.                      # []        Perfectly expanded bypass exhaust

# FUEL PROPERTIES
H = 120e6                       # [J/kg]    Fuel heat capacity for Hydrogen
fst = np.full(elements, 0.0291) # []        Stochiometric fuel-air ratio Hydrogen
philim = 1.                     # []        Maximum equivalence ratio possible in each stream

# INLET
taui = 1.                       # []        Inlet stagnation temperature ratio: Tti / Tt0
pidmax = 0.96                   # []        Maximum inlet pressure ratio due to frictional losses
Aratio = 2.5                    # []        Diffuser area ratio: A2 / A1

# CORE
ec = 0.91                       # []        Compressor polytropic efficiency
ef = 0.93                       # []        Fan polytropic efficiency
etab = 0.99                     # []        Burner combustion efficiency
etaAB = 0.99                    # []        Afterburner combustion efficiency
et = 0.93                       # []        Turbine polytropic efficiency
etam = 0.98                     # []        Mechanical transmission efficiency
pin = 0.95                      # []        Nozzle stagnation pressure ratio: Pt9 / Pt7
taun = 1.                       # []        Nozzle stagnation temperature ratio - adiabatic: Tt9 / Tt7
Tt4max = 2000.                  # [K]       Maximum turbine entry stagnation temperature limit

# BYPASS
pifn = 0.95                     # []        Bypass nozzle stagnation pressure ratio: Pt19 / Pt14
taufn = 1.                      # []        Bypass nozzle stagnation temperature ratio - adiabatic: Tt19 / Tt14
etaB = 0.99                     # []        Bypass burner combustion efficiency
Mbypassburn = 3.                # []        Miniumum flight mach number required for stable combustion in the bypass burner

#%% PERFORMANCE EQUATIONS
def solve_specific_thrust(u9a0: arr, u19a0: arr, M0: arr, alpha: float) -> arr:
    # Compute specific thrust at range of flight speeds
    ST = a0 / (1 + alpha) * ((u9a0 - M0) + alpha * (u19a0 - M0))
    return ST

def solve_specific_fuel_consumption(f: arr, fAB: arr, fB: arr, ST: arr, alpha: float) -> arr:
    SFC = ((f + fAB) / (1 + alpha) + fB / (1 + 1 / alpha)) / ST
    return SFC

def solve_uea0(TeT0: float, Mesqr: float) -> arr:
    # TeT0:     T9 / T0 or T19 / T0
    # Mesqr:    M9 ** 2 or M19 ** 2
    # uea0:     u9 / a0 or u19 / a0
    uea0 = np.sqrt((kt * Rt) / (kc * Rc) * TeT0 * Mesqr)
    #uea0 = (kt * Rt) / (kc * Rc) * TeT0 * Mesqr # Alternate equation
    np.nan_to_num(uea0, copy=False)
    return uea0

def solve_Mmax() -> float:
    Mmax = np.sqrt((Tt4max / T0 - 1) * 2 / (kc - 1))
    return Mmax

#%% INLET EQUATIONS
def solve_mass_flow_paramter(M) -> arr:
    MFP = np.sqrt(kc / Rc) * M * (1 + (kc - 1) / 2 * np.square(M)) ** (-1. * (kc + 1) / (2 * (kc - 1)))
    return MFP

def minimise_M2_error(M2, MFPM1):
    MFPM2 = solve_mass_flow_paramter(M2)
    err = Aratio - (MFPM1 / MFPM2)
    return err

def solve_M1(M0: arr) -> arr:
    # Solve post normal shock M (pre-diffuser). May be considering twice
    M0sqr = np.square(M0)
    M1 = np.sqrt(((kc - 1) * M0sqr + 2) / (2 * kc * M0sqr - (kc - 1)))
    return M1

def solve_M2(M0: arr) -> arr:
    if False:
        M1 = solve_M1(M0)
    else:
        M1 = M0
    MFPM1 = solve_mass_flow_paramter(M1)
    M2guess = M1 * 2
    M2, info, ier, mesg = fsolve(minimise_M2_error, x0=M2guess, args=(MFPM1), full_output=True)
    if ier != 1:
        raise Exception(f"{mesg}") # Error if fsolve did not converge
    if False:
        M2 = fsolve(lambda M: MFP(M,kc,Rc) - MFP(M1,kc,Rc) / Aratio, M2guess) # Alternate equation
    return M2

def solve_pii(M0: arr) -> arr:
    if False: # Conditions for wider range of M
        pii = np.zeros_like(M0)
        etar = np.zeros_like(M0)
        # Define conditions for etar
        case1 = M0 <= 1.
        case2 = M0 < 5.
        case3 = M0 >= 5.
        # Define etar values for each condition
        etar1 = np.full_like(M0, 1.)
        etar2 = 1. - 0.075 * (M0 - 1.) ** 1.35
        etar3 = 800. / (np.square(np.square(M0)) + 985.)
        # Set etar values based on condition
        etar[case1] = etar1[case1]
        etar[case2] = etar2[case2]
        etar[case3] = etar3[case3]
    else:
        etar = 1. - 0.075 * (M0 - 1.) ** 1.35
    pii = pidmax * etar
    return pii

def MFP(M,k,R): # Vince's equation for MFP. Mine is faster
    return np.sqrt(k/R)*M*(1.0 + 0.5*(k - 1.0)*M**2)**(-0.5*(k + 1.0)/(k - 1.0))

#%% CORE FLOW EQUATIONS
def solve_T9T0(tauAB: arr, taulambda: arr, taut: arr, pit: arr, pii: arr,
               pi0: arr, pic: float, pib: float, piAB: float) -> arr:
    T9T0 = taun * tauAB * taut * taulambda / (P0P9 * pin * piAB * pit * pib * pic * pii * pi0) ** ((kt - 1) / kt) * cpc / cpt
    np.nan_to_num(T9T0, copy=False)
    return T9T0

def solve_M9sqr(pit: arr, pii: arr, pi0: arr, pic: float, pib: float, piAB: float) -> arr:
    M9sqr = 2 / (kt - 1) * ((P0P9 * pin * piAB * pit * pib * pic * pii * pi0) ** ((kt - 1) / kt) - 1)
    return M9sqr

def solve_tauc(pic: float) -> arr:
    tauc = np.full(elements, pic ** ((kc - 1) / (kc * ec)))
    return tauc

def solve_tau0(M0: arr) -> arr:
    tau0 = 1 + (kc - 1) / 2 * np.square(M0)
    return tau0

def solve_taulambda() -> arr:
    taulambda = np.full(elements, (cpt * Tt4max) / (cpc * T0))
    return taulambda

def solve_f(taulambda: arr, tauc: arr, tau0: arr) -> arr:
    f = (taulambda - tauc * taui * tau0) / ((etab * H) / (cpc * T0) - taulambda)
    f[f < 0] = 0 # Set negative f values to 0
    if np.any(f < 0):
        raise Exception("f contains negatives")
    return f

def solve_tauf(pif: float) -> arr:
    tauf = np.full(elements, pif ** ((kc - 1) / (kt * ef)))
    return tauf

def solve_taut(tau0: arr, taulambda: arr, f: arr, tauc: arr, tauf, alpha: float) -> arr:
    taut = 1 - tau0 * taui / (taulambda * etam * (1 + f)) * (tauc - 1 + alpha * (tauf - 1))
    return taut

def solve_pit(taut: arr) -> arr:
    pit = taut ** (kt / (kt - 1) * et)
    np.seterr(all='warn')
    np.nan_to_num(pit, copy=False)
    return pit

def solve_fAB(f: arr) -> arr:
    fAB = fst - f
    return fAB

def solve_tauAB(f: arr, fAB: arr, taut: arr, taulambda: arr, tau0: arr) -> arr:
    tauAB = 1 / (1 + f + fAB) * (1 + f + (etaAB * fAB * H) / (taut * taulambda * tau0 * T0 * cpc))
    return tauAB

def solve_pi0(M0: arr) -> arr:
    pi0 = (1 + (kc - 1) * 0.5 * np.square(M0)) ** (kc / (kc - 1))
    return pi0

#%% BYPASS BURNER EQUATIONS
def solve_T19T0(tauB: arr, tau0: arr, tauf: arr, piB: arr, pii: arr, pi0: arr, pif: float) -> arr:
    #T19T0 = taun * tauB * tauf * taui * tau0 / (P0P19 * pin * piB * pif * pii * pi0) ** ((kt + 1) / kt) # Alternate equation
    T19T0 = taun * tauB * tauf * taui * tau0 / (P0P19 * pin * piB * pif * pii * pi0) ** ((kc + 1) / kc)
    return T19T0

def solve_M19sqr(piB: arr, pii: arr, pi0: arr, pif: float) -> arr:
    #M19sqr = 2 / (kt - 1) * ((P0P19 * pin * piB * pif * pii * pi0) ** ((kt - 1) / kt) - 1) # Alternate Equation
    M19sqr = 2 / (kc - 1) * ((P0P19 * pin * piB * pif * pii * pi0) ** ((kc - 1) / kc) - 1)
    return M19sqr

def solve_tauBmax(M13: arr) -> arr: # Choke
    M13sqr = np.square(M13)
    tauBmax = np.square(1 + kc * M13sqr) / (2 * (kc + 1) * M13sqr * (1 + (kc - 1) / 2 * M13sqr))
    return tauBmax

def solve_phimax(tau0: arr, tauBmax: arr, tauf: arr) -> arr:
    #phimax = cpc * T0 * tau0 * tauf * (tauBmax - 1) / (fst * H) # Alternate equation
    phimax = cpc * tauf * taui * tau0 * T0 * (tauBmax - 1) / (etaB * fst * H)
    #phimax = cpc * taui * tau0 * tauf * T0 * (tauBmax - 1) / (fst * (etaB * H - cpc * tauBmax * tauf * taui * tau0 * T0)) # Alternate equation
    if np.any(phimax > 1):
        raise Exception("phimax greater than 1")
    return phimax

def solve_phi(M13: arr, tau0: arr, tauf: arr) -> arr:
    tauBmax = solve_tauBmax(M13)
    phimax = solve_phimax(tau0, tauBmax, tauf)
    phi = np.minimum(phimax, 1.)
    return phi

def solve_tauB(tau0: arr, tauf: arr, phi: arr) -> arr: # Thermal choke
    #tauB = etaB * phi * fst * H / (cpc * T0 * tau0 * tauf) + 1 # Alternate equation
    #tauB = etaB * phi * fst * H / (cpc * tauf * taui * tau0 * T0) + 1 # Alternate equation
    #tauB = (etaB * phi * fst * H + cpc * tauf * taui * tau0 * T0) / ((phi * fst + 1) * cpc * tauf * taui * tau0 * T0) # Alternate equation
    tauB = (etaB * phi * fst * H / (tau0 * T0 * cpc) + tauf) / (tauf * (phi * fst + 1))
    return tauB 

def minimise_M14_err(M14: arr, M13sqr: arr, tauB: arr) -> arr:
    M14sqr = np.square(M14)
    #err = tauB - (1 + ((kt - 1) * 0.5 * M14sqr)) / (1 + ((kc - 1) * 0.5 * M13sqr)) * M14sqr / M13sqr * np.square((1 + kc * M13sqr) / (1 + kt * M14sqr)) # Alternate equation
    err = tauB - (1 + ((kc - 1) * 0.5 * M14sqr)) / (1 + ((kc - 1) * 0.5 * M13sqr)) * M14sqr / M13sqr * np.square((1 + kc * M13sqr) / (1 + kc * M14sqr))
    if False:
        print([err], flush=True)
    return err

def solve_M14(M13: arr, tauB: arr) -> arr:
    M13sqr = np.square(M13)
    if False: # Numerically solve M14
        M14guess = M13 * 1.
        M14, info, ier, mesg = fsolve(minimise_M14_err, x0=M14guess, args=(M13sqr, tauB), full_output=True)
        if ier != 1:
            raise Exception(f"{mesg}") # Error if fsolve did not converge
    elif True: # Quadratic formula for M14
        A = tauB * (1 + (kc - 1) * 0.5 * M13sqr) * M13sqr / np.square(1 + kc * M13sqr)
        a = (kc - 1) * 0.5 - A * kc * kc
        b = 1 - 2 * kc * A
        c = -1. * A
        # M14sqrminus = (-1. * b - np.sqrt(np.square(b) - 4 * a * c)) / (2 * a) # -ve invalid
        M14sqrplus = (-1. * b + np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)
        M14 = np.sqrt(M14sqrplus)
    else: # Formula sheet
        chi = tauB * M13sqr * (1 + (kc - 1) * 0.5 * M13sqr) / (np.square(1 + kc * M13sqr))
        M14sqr = 2 * chi / (1 - 2 * chi * kc + np.sqrt(1 - 2 * chi * (kc + 1)))
        M14 = np.sqrt(M14sqr)
    return M14

def solve_piB(M13: arr, M14: arr) -> arr:
    M13sqr = np.square(M13)
    M14sqr = np.square(M14)
    #piB = (1 + kc * M13sqr) / (1 + kt * M14sqr) * ((1 + (kt - 1) * 0.5 * M14sqr) ** (kt / (kt - 1))) / ((1 + (kc - 1) * 0.5 * M13sqr) ** (kc / (kc - 1))) # Alternate equation
    piB = (1 + kc * M13sqr) / (1 + kc * M14sqr) * ((1 + (kc - 1) * 0.5 * M14sqr) ** (kc / (kc - 1))) / ((1 + (kc - 1) * 0.5 * M13sqr) ** (kc / (kc - 1)))
    return piB

def solve_fB(phi: arr) -> arr:
    fB = phi * fst
    return fB

#%% MODEL SOLVER INTERFACE
def calculate_model(mode: int, M0: arr, pic: float, pif: float, alpha: float) -> arr:
    # mode: 1 = pure turbofan
    #       2 = afterburning turbofan
    #       3 = turboramjet
    #       4 = ramjet
    if mode == 1:
        ST, SFC = calculate_turbofan(M0=M0, pic=pic, pif=pif, alpha=alpha)
    elif mode == 2:
        ST, SFC = calculate_afterburning_turbofan(M0=M0, pic=pic, pif=pif, alpha=alpha)
    elif mode == 3:
        ST, SFC = calculate_turboramjet(M0=M0, pic=pic, pif=pif, alpha=alpha)
    elif mode == 4:
        ST, SFC = calculate_ramjet(M0=M0, pic=pic, pif=pif, alpha=alpha)
    else:
        raise Exception("Invalid mode, choose from: 1 = pure turbofan, 2 = afterburning turbofan, 3 = turboramjet, 4 = ramjet")
    return ST, SFC

def calculate_turbofan(M0: arr, pic: float, pif: float, alpha: float) -> arr:
    # Inlet
    M2 = solve_M2(M0)
    # Core
    pib = 0.99
    piAB = 1.
    taulambda = solve_taulambda()
    tauc = solve_tauc(pic)
    tau0 = solve_tau0(M0)
    tauf = solve_tauf(pif)
    f = solve_f(taulambda, tauc, tau0)
    fAB = np.zeros(elements) # No fuel injection in afterburner
    taut = solve_taut(tau0, taulambda, f, tauc, tauf, alpha)
    tauAB = np.ones(elements)
    taulambda = solve_taulambda()
    pit = solve_pit(taut)
    pii = solve_pii(M0)
    pi0 = solve_pi0(M0)
    T9T0 = solve_T9T0(tauAB, taulambda, taut, pit, pii, pi0, pic, pib, piAB)
    M9sqr = solve_M9sqr(pit, pii, pi0, pic, pib, piAB)
    u9a0 = solve_uea0(T9T0, M9sqr)
    # Bypass
    M13 = M2
    phi = solve_phi(M13, tau0, tauf) # Not used
    tauB = np.ones(elements)
    M14 = solve_M14(M13, tauB) # Not used
    #piB = solve_piB(M13, M14)
    piB = np.ones(elements)
    T19T0 = solve_T19T0(tauB, tau0, tauf, piB, pii, pi0, pif)
    M19sqr = solve_M19sqr(piB, pii, pi0, pif)
    u19a0 = solve_uea0(T19T0, M19sqr)
    fB = np.zeros(elements) # No fuel injection in bypass burner
    # Performance
    ST = 2 * solve_specific_thrust(u9a0, u19a0, M0, alpha)
    SFC = solve_specific_fuel_consumption(f, fAB, fB, ST, alpha)
    # Plot
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(M0, pii, label="pii")
    axs[0].plot(M0, pit, label="pit")
    axs[0].plot(M0, piB, label="piB")
    axs[0].legend()
    axs[0].set_xlabel("M")
    axs[0].set_title("pi")
    axs[1].plot(M0, tauc, label="tauc")
    axs[1].plot(M0, tau0, label="tau0")
    axs[1].plot(M0, tauf, label="tauf")
    axs[1].plot(M0, taut, label="taut")
    axs[1].plot(M0, tauB, label="tauB")
    axs[1].plot(M0, taulambda, label="taulambda")
    axs[1].plot(M0, tauAB, label="tauAB")
    axs[1].legend()
    axs[1].set_xlabel("M")
    axs[1].set_title("tau")
    axs[2].plot(M0, f, label="f")
    axs[2].plot(M0, fAB, label="fAB")
    axs[2].plot(M0, fst, label="fst")
    axs[2].plot(M0, fB, label="fB")
    axs[2].legend()
    axs[2].set_xlabel("M")
    axs[2].set_title("f")
    fig.suptitle("turbofan")
    plt.show()
    return ST, SFC

def calculate_afterburning_turbofan(M0: arr, pic: float, pif: float, alpha: float) -> arr:
    # Inlet
    M2 = solve_M2(M0)
    # Core
    pib = 0.99
    piAB = 0.99
    taulambda = solve_taulambda()
    tauc = solve_tauc(pic)
    tau0 = solve_tau0(M0)
    tauf = solve_tauf(pif)
    f = solve_f(taulambda, tauc, tau0)
    fAB = solve_fAB(f)
    taut = solve_taut(tau0, taulambda, f, tauc, tauf, alpha)
    tauAB = solve_tauAB(f, fAB, taut, taulambda, tau0)
    taulambda = solve_taulambda()
    pit = solve_pit(taut)
    pii = solve_pii(M0)
    pi0 = solve_pi0(M0)
    T9T0 = solve_T9T0(tauAB, taulambda, taut, pit, pii, pi0, pic, pib, piAB)
    M9sqr = solve_M9sqr(pit, pii, pi0, pic, pib, piAB)
    u9a0 = solve_uea0(T9T0, M9sqr)
    # Bypass
    M13 = M2
    phi = solve_phi(M13, tau0, tauf) # Not used
    tauB = np.ones(elements) # Bypass burner off
    M14 = solve_M14(M13, tauB) # not ysed
    piB = np.ones(elements) # Bypass burner off
    T19T0 = solve_T19T0(tauB, tau0, tauf, piB, pii, pi0, pif)
    M19sqr = solve_M19sqr(piB, pii, pi0, pif)
    u19a0 = solve_uea0(T19T0, M19sqr)
    fB = np.zeros(elements) # Bypass burner fuel off
    # Performance
    ST = 2 * solve_specific_thrust(u9a0, u19a0, M0, alpha)
    SFC = solve_specific_fuel_consumption(f, fAB, fB, ST, alpha)
    # Plot
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(M0, pii, label="pii")
    axs[0].plot(M0, pit, label="pit")
    axs[0].plot(M0, piB, label="piB")
    axs[0].legend()
    axs[0].set_xlabel("M")
    axs[0].set_title("pi")
    axs[1].plot(M0, tauc, label="tauc")
    axs[1].plot(M0, tau0, label="tau0")
    axs[1].plot(M0, tauf, label="tauf")
    axs[1].plot(M0, taut, label="taut")
    axs[1].plot(M0, tauB, label="tauB")
    axs[1].plot(M0, taulambda, label="taulambda")
    axs[1].plot(M0, tauAB, label="tauAB")
    #axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_xlabel("M")
    axs[1].set_title("tau")
    axs[2].plot(M0, f, label="f")
    axs[2].plot(M0, fAB, label="fAB")
    axs[2].plot(M0, fst, label="fst")
    axs[2].plot(M0, fB, label="fB")
    axs[2].legend()
    axs[2].set_xlabel("M")
    axs[2].set_title("f")
    fig.suptitle("afterburning turbofan")
    plt.show()
    return ST, SFC

def calculate_turboramjet(M0: arr, pic: float, pif: float, alpha: float) -> arr:
    # Inlet
    M2 = solve_M2(M0)
    # Core
    pib = 0.99
    piAB = 0.99
    taulambda = solve_taulambda()
    tauc = solve_tauc(pic)
    tau0 = solve_tau0(M0)
    tauf = solve_tauf(pif)
    f = solve_f(taulambda, tauc, tau0)
    fAB = solve_fAB(f)
    taut = solve_taut(tau0, taulambda, f, tauc, tauf, alpha)
    tauAB = solve_tauAB(f, fAB, taut, taulambda, tau0)
    taulambda = solve_taulambda()
    pit = solve_pit(taut)
    pii = solve_pii(M0)
    pi0 = solve_pi0(M0)
    T9T0 = solve_T9T0(tauAB, taulambda, taut, pit, pii, pi0, pic, pib, piAB)
    M9sqr = solve_M9sqr(pit, pii, pi0, pic, pib, piAB)
    u9a0 = solve_uea0(T9T0, M9sqr)
    # Bypass
    M13 = M2
    phi = solve_phi(M13, tau0, tauf)
    if considerMbypassburn: # True if factoring in Mbypassburn
        tauB = np.ones(elements)                                               # Set tauB to 1 until Mbypass is reached
        tauB[M0 > Mbypassburn] = solve_tauB(tau0, tauf, phi)[M0 > Mbypassburn] # Bypass burner only after Mbypassburn
    else:
        tauB = solve_tauB(tau0, tauf, phi)                                     # Bypass burner always on
    M14 = solve_M14(M13, tauB)
    if considerMbypassburn: # True if factoring in Mbypassburn
        piB = np.ones(elements)                                                # Set piB to 1 until Mbypassburn is reached
        piB[M0 > Mbypassburn] = solve_piB(M13, M14)[M0 > Mbypassburn]          # Bypass burner only after Mbypassburn
    else:
        piB = solve_piB(M13, M14)                                              # Bypass burner always on
    T19T0 = solve_T19T0(tauB, tau0, tauf, piB, pii, pi0, pif)
    M19sqr = solve_M19sqr(piB, pii, pi0, pif)
    u19a0 = solve_uea0(T19T0, M19sqr)
    fB = solve_fB(phi)
    # Performance
    ST = 2 * solve_specific_thrust(u9a0, u19a0, M0, alpha)
    SFC = solve_specific_fuel_consumption(f, fAB, fB, ST, alpha)
    
    M1 = solve_M1(M0)                                                          # Used for plotting to verify MFP calculation

    # Plot
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(M0, pii, label="pii")
    axs[0].plot(M0, pit, label="pit")
    axs[0].plot(M0, piB, label="piB")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel("M")
    axs[0].set_title("pi")
    axs[1].plot(M0, tauc, label="tauc")
    axs[1].plot(M0, tau0, label="tau0")
    axs[1].plot(M0, tauf, label="tauf")
    axs[1].plot(M0, taut, label="taut")
    axs[1].plot(M0, tauB, label="tauB")
    axs[1].plot(M0, taulambda, label="taulambda")
    axs[1].plot(M0, tauAB, label="tauAB")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_xlabel("M")
    axs[1].set_title("tau")
    axs[2].plot(M0, f, label="f")
    axs[2].plot(M0, fAB, label="fAB")
    axs[2].plot(M0, fst, label="fst")
    axs[2].plot(M0, fB, label="fB")
    axs[2].grid()
    axs[2].legend()
    axs[2].set_xlabel("M")
    axs[2].set_title("f")
    fig.suptitle("turboramjet")
    
    fig2, axs2 = plt.subplots(1, 1)
    axs2.plot(M0, phi, label="phi")
    axs2.legend()
    axs2.set_xlabel("M0")
    axs2.set_title("phi")
    
    fig3, axs3 = plt.subplots(1, 1)
    axs3.plot(M0, M13, label="M13 = M2")
    axs3.plot(M0, M14, label="M14")
    axs3.plot(M0, M1, label="M1")                                              # Verify MFP calculation
    axs3.grid()
    axs3.legend()
    axs3.set_xlabel("M")
    axs3.set_title("bypass burner M - turboramjet")
    
    plt.show()
    return ST, SFC

def calculate_ramjet(M0: arr, pic: float, pif: float, alpha: float) -> arr:
    # Inlet
    M2 = solve_M2(M0)
    # Core
    pib = 1.
    piAB = 0.99
    taulambda = solve_taulambda()
    tauc = np.ones(elements)                                                   # Compressor off
    tau0 = solve_tau0(M0)
    tauf = np.ones(elements)                                                   # Fan off
    f = np.zeros(elements)                                                     # Main burner fuel off
    fAB = np.full(elements, fst)                                               # All core fuel injected in afterburner
    taut = np.ones(elements)                                                   # Turbine off
    tauAB = solve_tauAB(f, fAB, taut, taulambda, tau0)
    taulambda = solve_taulambda()
    pit = np.ones(elements)                                                    # Tubrine off -> no pressure change
    pii = solve_pii(M0)
    pi0 = solve_pi0(M0)
    T9T0 = solve_T9T0(tauAB, taulambda, taut, pit, pii, pi0, pic, pib, piAB)
    M9sqr = solve_M9sqr(pit, pii, pi0, pic, pib, piAB)
    u9a0 = solve_uea0(T9T0, M9sqr)
    # Bypass
    M13 = M2
    phi = solve_phi(M13, tau0, tauf)
    if considerMbypassburn: # True if factoring in Mbypassburn
        tauB = np.ones(elements)                                               # Set tauB to 1 until Mbypassburn is reached
        tauB[M0 > Mbypassburn] = solve_tauB(tau0, tauf, phi)[M0 > Mbypassburn] # Bypass burner only after Mbypassburn
    else:
        tauB = solve_tauB(tau0, tauf, phi)                                     # Bypass burner always on
    M14 = solve_M14(M13, tauB)
    if considerMbypassburn: # True if factoring in Mbypassburn
        piB = np.ones(elements)                                                # Set piB to 1 until Mbypass is reached
        piB[M0 > Mbypassburn] = solve_piB(M13, M14)[M0 > Mbypassburn]          # Bypass burner only after Mbypass
    else:
        piB = solve_piB(M13, M14)                                              # Bypass burner always on
    T19T0 = solve_T19T0(tauB, tau0, tauf, piB, pii, pi0, pif)
    M19sqr = solve_M19sqr(piB, pii, pi0, pif)
    u19a0 = solve_uea0(T19T0, M19sqr)
    fB = solve_fB(phi)
    # Performance
    ST = 2 * solve_specific_thrust(u9a0, u19a0, M0, alpha)
    SFC = solve_specific_fuel_consumption(f, fAB, fB, ST, alpha)
    # Plot
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(M0, pii, label="pii")
    axs[0].plot(M0, pit, label="pit")
    axs[0].plot(M0, piB, label="piB")
    axs[0].legend()
    axs[0].set_xlabel("M")
    axs[0].set_title("pi")
    axs[1].plot(M0, tauc, label="tauc")
    axs[1].plot(M0, tau0, label="tau0")
    axs[1].plot(M0, tauf, label="tauf")
    axs[1].plot(M0, taut, label="taut")
    axs[1].plot(M0, tauB, label="tauB")
    axs[1].plot(M0, taulambda, label="taulambda")
    axs[1].plot(M0, taut, label="taut")
    axs[1].plot(M0, tauAB, label="tauAB")
    axs[1].legend()
    axs[1].set_xlabel("M")
    axs[1].set_title("tau")
    axs[2].plot(M0, f, label="f")
    axs[2].plot(M0, fAB, label="fAB")
    axs[2].plot(M0, fst, label="fst")
    axs[2].plot(M0, fB, label="fB")
    axs[2].legend()
    axs[2].set_xlabel("M")
    axs[2].set_title("f")
    fig.suptitle("ramjet")

    fig2, axs2 = plt.subplots(1, 1)
    axs2.plot(M0, phi, label="phi")
    axs2.legend()
    axs2.set_xlabel("M0")
    axs2.set_title("phi")
    
    fig3, axs3 = plt.subplots(1, 1)
    axs3.plot(M0, M13, label="M13")
    axs3.plot(M0, M14, label="M14")
    axs3.legend()
    axs3.set_xlabel("M")
    axs3.set_title("bypass burner M - ramjet")
    
    plt.show()
    return ST, SFC

#%% MAIN
def main() -> None:
    start = time.time()
    
    M0 = np.linspace(1, 5, elements)
    
    ST1, SFC1 = calculate_model(mode=1, M0=M0, pic=30., pif=2., alpha=1.)
    ST2, SFC2 = calculate_model(mode=2, M0=M0, pic=30., pif=2., alpha=1.)
    ST3, SFC3 = calculate_model(mode=3, M0=M0, pic=30., pif=2., alpha=1.)
    ST4, SFC4 = calculate_model(mode=4, M0=M0, pic=1.,  pif=1., alpha=1.)

    fig1, axs1 = plt.subplots(1, 2)

    axs1[0].plot(M0, ST1, label="turbofan")
    axs1[0].plot(M0, ST2, label="afterburning turbofan")
    axs1[0].plot(M0, ST3, label="turboramjet")
    axs1[0].plot(M0, ST4, label="ramjet")
    axs1[0].grid()
    axs1[0].legend()
    axs1[0].set_xlabel("M0")
    axs1[0].set_title("ST")

    axs1[1].plot(M0, SFC1, label="turbofan")
    axs1[1].plot(M0, SFC2, label="afterburning turbofan")
    axs1[1].plot(M0, SFC3, label="turboramjet")
    axs1[1].plot(M0, SFC4, label="ramjet")
    axs1[1].grid()
    axs1[1].legend()
    axs1[1].set_xlabel("M0")
    axs1[1].set_title("SFC")
        
    plt.show()
    
    Mmax = solve_Mmax()
    print(f"\nMmax = {Mmax:.2f}")
    
    finish = time.time()
    
    print(f"\nExecution time: {finish - start:.3f}s")
    
    return

if __name__ == '__main__':
    main()