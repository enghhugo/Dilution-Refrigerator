import numpy as np
import scipy.optimize as sp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as ip


#                                   Initialisering

L_d = 1  # Längd i meter
L_c = 3  # Längd i meter
# Mängd sektioner, fler sektioner gör uträkningen långsammare, men ger slätare resultat
N = int(1500)
dL_d = L_d / N  # Längd per stycke
dL_c = L_c / N  # Längd per stycke
L_pos_d = np.linspace(0, L_d, N)
L_pos_c = np.linspace(0, L_c, N)
T_mc = 0.020
x_mc = 0.066  # Koncentration i mixing chamber
m_punkt = 600e-6  # flöde i mol
r = 0.002 # radie i meter
p = 2*np.pi*r  # omkrets i meter 
Z_d = 8 * L_d / (np.pi*r**4)  # Impedans enligt Hagen-Pausille
Z_c = 8 * L_c / (np.pi*r**4)  # Impedans enligt Hagen-Pausille


#                                 Händiga funktioner
# Funktioner för kemisk potential:
def mu_40_get(T):
    T_vector = np.array([0.000, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030, 0.035,
                        0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.10, 0.120, 0.140, 0.160, 0.180, 0.20, 0.250, 0.300, 0.350, 0.40, 0.450, 0.50])
    mu_vector = np.array([0, 6.808e-15, 1.089e-13, 5.514e-13, 1.743e-12,  8.823e-12,  2.789e-11,  6.808e-11, 1.412e-10,  2.615e-10,  4.462e-10,  7.147e-10,  1.089e-9, 2.659e-9, 5.514e-9, 1.022e-8,
                         1.743e-8, 2.792e-8, 4.255e-8, 8.823e-8, 1.635e-7,  2.789e-7, 4.467e-7, 6.808e-7, 1.412e-6, 2.615e-6, 4.462e-6, 7.147e-6, 1.089e-5, 2.659e-5, 5.514e-5, 1.022e-4, 1.743e-4, 2.792e-4, 4.255e-4])
    # Tabell 5 i Radebaugh, kemiska potential är tagen mot T=0 som referenspkt
    mu_40 = np.interp(T, T_vector, mu_vector)
    return mu_40  # [J/mol]


def PiV_get(T, x, matrix):
    matrix = matrix.astype(float)
    x_vector = np.array(matrix[0, 1:])
    T_vector = np.array(matrix[1:, 0])
    # Interpolerar över grafen, gör den slät
    Interpol = ip.RectBivariateSpline(T_vector, x_vector, matrix[1:, 1:])
    PiV = Interpol(T, x)
    return PiV[0, 0]  # [J/mol]


def mu_4_get(T, x, matrix):
    mu_4 = mu_40_get(T) - PiV_get(T, x, matrix)  # Källa 1 ekv 2
    return mu_4  # [J/mol]


def residual(x, T_matris, mu_4_mc, i, matrix):
    return mu_4_get(T_matris[1, i], x, matrix) - mu_4_mc


# Funktioner för värmeöverföring

def V_get(x):
    V_cm3 = (27.58 - 3.3 * x ** 3)  # Källa 1 ekv 3
    V = V_cm3 * 1e-6  # Notera cm3 till m3 factorn
    return V  # [m^3/mol]


def Qvisc_get(T, x, Z, sida):

    # 1) Volymflöde (m^3/s) från dina funktioner
    V_punkt = V_get(x) * m_punkt

    # 2) Viskositet (Pa*s)
    if sida == 1:  # logik som säger om du räknar på den koncentrerade sidan eller den utspädda sidan
        eta = 1.81e-6 / T**2  # conc
    else:
        eta = 510e-7 / T**2  # dilute

    # 3) dP/dx för laminärt flöde i rör enligt Hagen-Poiseuille: dP/dx = (8 * eta * V) / (pi * r^4)
    dPdL = Z * eta * V_punkt   # Pa/m

    # 4) Viskös uppvärmning (effekt) i sektionen: Qvisc = ΔP * Q
    Qvisc = dPdL * V_punkt                          # W/m

    return Qvisc


def R_kt_get(T_c, T_d):
    if T_c < 0.13:
        # Uträknat i SI-Enheter från ekvation 2 källa ekvation 2a, [m^2*K/W]
        rho_c = 20e-3
    else:
        # Uträknat i SI-Enheter från ekvation 2 källa ekvation 2b, [m^2*K/W]
        rho_c = (2.4/(T_c)+1.55)*1e-3
    # Uträknat i SI-Enheter från ekvation 2 källa ekvation 2b, [m^2*K/W]
    rho_d = 7e-3
    return rho_c+rho_d


def dTcdx_radial_get(T_c, T_d):
    dTcdx_radial = ((1/88)*p/T_c)*(T_d**4 - T_c**4) / \
        (m_punkt*R_kt_get(T_c, T_d))  # ekvation 13 källa 1
    return dTcdx_radial


def dTddx_radial_get(T_c, T_d):
    dTddx_radial = 22*T_c/(106*T_d)*dTcdx_radial_get(T_c,
                                                     T_d)  # ekvation 14 källa 1
    return dTddx_radial


def dTcdx_get(T_c, T_d, x):
    dTcdx_visc = Qvisc_get(T_c, x, Z_c, 1)/(m_punkt*22*T_c)
    dTcdx = dTcdx_radial_get(T_c, T_d) + dTcdx_visc
    return dTcdx


def dTddx_get(T_c, T_d, x):
    dTddx_visc = Qvisc_get(T_d, x, Z_d, 0)/(m_punkt*22*T_c)
    dTddx = dTddx_radial_get(T_c, T_d) + dTddx_visc

    return dTddx


def VVX_Loop(x, plotta, matrix):
    T_guess = x  # Heter så pga brentq
    mu_4_mc = mu_4_get(T_mc, x_mc, matrix)
    # rad 1 är koncentrerade sidan, rad 2 är utspädda, kolonnerna är stegen från MC till still.
    T_matris = np.zeros((2, N))
    x_matris = np.zeros((2, N))  # pss som ovan
    T_matris[0, N-1] = 0.7  # T_kond
    x_matris[0, :] = 1  # Koncentrationen är 1 på koncentrerade sidan
    T_matris[1, N-1] = T_guess  # Bestämmer gissning av T1d
    for i in range(N-1, 0, -1):
        # print(T_matris[0, i])
        # print(T_matris[1, i])
        # finner x_id s.a. mu_4i = mu_4mc
        x_matris[1, i] = sp.brentq(
            residual, 0.0, 0.12, xtol=0.001, args=(T_matris, mu_4_mc, i, matrix))
        T_matris[0, i-1] = T_matris[0, i] + \
            dTcdx_get(T_matris[0, i], T_matris[1, i],
                      x_matris[0, i])*dL_c  # DeltaT per steg
        T_matris[1, i-1] = T_matris[1, i] + \
            dTddx_get(T_matris[0, i], T_matris[1, i], x_matris[1, i])*dL_d
        if T_matris[1, i] < 0:
            T_matris[1, i-1] = T_matris[1, i]
        elif T_matris[1, i] < T_matris[1, i-1]:
            T_matris[1, i-1] = T_matris[1, i]

    if plotta == 1:
        # Plotta
        plt.figure()
        plt.plot(L_pos_c, T_matris[0, :],
                 label='T_c (concentrated side)', color='red')
        plt.plot(L_pos_c, T_matris[1, :],
                 label='T_d (dilute side)', color='blue')
        # Pil för varm ström
        plt.annotate(
            '',
            xy=(0.40, T_matris[0, :][int(0.70*len(L_pos_c))]),
            xytext=(0.55, T_matris[0, :][int(0.79*len(L_pos_c))]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )

        # Pil för kall ström
        plt.annotate(
            '',
            xy=(0.4, T_matris[1, :][int(0.65*len(L_pos_d))]),
            xytext=(0.25, T_matris[1, :][int(0.57*len(L_pos_d))]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2)
        )
        plt.xlabel("Position x [m]")
        plt.ylabel("Temperature [K]")
        plt.title("Counterflow HEX (In-House Solution)")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(L_pos_d[0:N-1], x_matris[1, 1:], color='blue')
        # Pil för varm ström
        plt.annotate(
            '',
            xy=(0.55, x_matris[1, 1:][int(0.65*len(L_pos_d[0:79]))]),
            xytext=(0.40, x_matris[1, 1:][int(0.45*len(L_pos_d[0:79]))]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2)
        )
        plt.xlabel("Position x [m]")
        plt.ylabel("x_d")
        plt.title("Counterflow HEX (In-House Solution)")
        plt.show()

        plt.figure()
        plt.plot(L_pos_c, x_matris[0, :], color='red')
        plt.xlabel("Position x [m]")
        plt.ylabel("x_c")
        plt.title("Counterflow HEX (In-House Solution)")
        plt.show()
        return T_matris[0, 0]
    elif T_matris[1, 0] - T_mc < 0:
        return -1
    else:
        return T_matris[1, 0] - T_mc

#                                     Huvudloop


def brent_VVX():
    PiV_file = '/Users/martinbornecrantz/Downloads/kemisk_potensial4(U4).csv'
    df = pd.read_csv(PiV_file, header=None)  # Importerar filen
    matrix = df.values  # Gör om den till matris
    T_in_i_still = sp.brentq(VVX_Loop, T_mc+0.001, 0.7, xtol=0.00001, args=(0, matrix))
    T_in_i_MC = VVX_Loop(T_in_i_still, 1, matrix)
    return T_in_i_MC, T_in_i_still


T_in_i_MC, T_in_i_still = brent_VVX()
print("färdig")
print(T_in_i_MC)
print(T_in_i_still)