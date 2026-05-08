"""
komponenter.py
==============
Fysikaliska komponentmodeller för utspädningskylarsimuleringen.

Varje klass representerar en fysisk komponent i DR-kretsen, förutom
`Fluid`-klassen som modellerar fluiden i rören.

Klasserna innehåller tabelldata från Radebaugh, komponentparametrar
och fysikaliska funktioner som beskriver deras beteende.

Komponentförteckning
--------------------
Blandningskammare — He-3 korsar fasgränsen här; kylan genereras
Stillkammare      — He-3-rik ånga pumpas bort härifrån
Pump              — Cirkulationspump
Värmeväxlare      — Motströmsvärmeväxlare (platshållare)
Fluid             — Beräknar tryckfall i rören från stillen till kondensorn
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import scipy.optimize as spo
import scipy.optimize as sp
import sympy
from scipy.optimize import least_squares
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


from .konstanter import R, gamma, k_b, M_He3, M_He4, m_He3, d_mol, sigma, r_vvx
from .termiskdata import TermiskData, standardtabeller


# ------------------------------------------------------------------
# Blandningskammaren
# ------------------------------------------------------------------

@dataclass
class Blandningskammare:
    """
    Modell för blandningskammaren.

    He-3 korsar fasgränsen från koncentrerad fas till utspädd fas.
    Den associerade entalpiskillnaden driver kylningen — det är här
    som kylan faktiskt genereras i utspädningskylaren.


    Parametrar
    ----------
    tabeller : TermiskData
        Termodynamiska uppslagstabeller. Använder standardinstansen
        om inget annat anges.
    """

    tabeller: TermiskData = field(default_factory=lambda: standardtabeller)

    def x_mc(self, T: float) -> float:
        """
        He-3-molfraktion i utspädd fas vid temperaturen T.

        Detta är ett polynom anpassat till fasdiagrammets gränskurva
        från Radebaugh. Kurvan beskriver hur mycket He-3 som löser sig
        i He-4 vid en given temperatur — ju högre temperatur, desto mer He-3.

            X_t(T) = 0.0648 · (1 + 8.4·T² + 9.4·T³)

        När molfraktionen når 6,6 % blir lösningen mättad.

        Parametrar
        ----------
        T : float  Temperatur  [K]

        Returvärde
        ----------
        X : float  He-3 molfraktion  [-]
        """

        x = 0.0648 * (1.0 + 8.4 * T**2 + 9.4 * T**3)

        if x > 0.066:
            return 0.066
        else: 
            return x
    

    def _H3_utspädd(self, T: float, tabeller: TermiskData = standardtabeller) -> float:
        """
        Molar entalpi för He-3 i utspädd fas, utvärderad vid löslighetskurvans
        sammansättning X_t(T).

        Kombinerar löslighetskurvan med entalpitabellen:

            H3_utspädd(T) = H_losning(T,  X_t(T))

        Det vill säga: vi rör oss alltid längs fasdiagrammets gränskurva
        när vi beräknar entalpiskillnaden i blandningskammaren.

        Parametrar
        ----------
        T        : float        Temperatur  [K]
        tabeller : TermiskData  Termodynamiska tabeller (valfri ersättning)

        Returvärde
        ----------
        float
            Entalpi  [J/mol]
        """
        return tabeller.H_losning_HE3_4(T, self.x_mc(T))

    def Q_allmän(
        self,
        m_prick: float,
        X_in: float,
        T_mc: float,
        T_in: float,
    ) -> float:
        """
        Kyleffekt för en blandad He-3/He-4-inloppsström.

        Härleds från en entalpibalans över blandningskammaren.
        En inloppsström med He-3-molfraktion X_in blandas med den
        utspädda fasen i blandningskammaren. He-3 passerar fasgränsen
        mellan koncentrerad och utspädd fas, vilket genererar kylning.

        Parametrar
        ----------
        m_prick : float
            Totalt molflöde för inloppsströmmen [mol/s]

        X_in : float
            He-3-molfraktion i inloppsströmmen [-]

        T_mc : float
            Blandningskammarens temperatur [K]

        T_in : float
            Temperaturen hos inkommande ström [K]

        Returvärde
        ----------
        Q : float 
         Kyleffekt  [W]

        """
        H_utspädd_mc  = self._H3_utspädd(T_mc, self.tabeller)
        H_ren_in      = self.tabeller.H_ren_HE3(T_in)
        H_utspädd_in  = self._H3_utspädd(T_in, self.tabeller)

        term_he3 = X_in * (H_utspädd_mc)
        term_he3_in = (X_in-self.x_mc(T_mc))*H_ren_in/(1-self.x_mc(T_mc))
        term_he4_in = self.x_mc(T_mc) * (1.0 - X_in) * H_utspädd_in/(1-self.x_mc(T_mc))

        return m_prick * (term_he3 - (term_he3_in + term_he4_in))

# ------------------------------------------------------------------
# Destillator
# ------------------------------------------------------------------

@dataclass
class Destillator:
    """
    Modell för class Destillatorkammare.

    He-3-rik ånga pumpas bort härifrån, vilket driver He-3-cirkulationen.
    Löser kemisk jämvikt med blandningskammaren för att hitta
    stilltemperatur T_s, molfracktionen X_s i gasen samt ångtrycket P_v.
    
    Löser ut molflödet utifrån destillatorn

    Parametrar
    ----------
    tabeller : TermiskData
        Termodynamiska uppslagstabeller.
        
        r: float   Diametern på destillatormynningen vid stillen. brukar vara runt 1 mm 
        rho_sup:   float kg/m^3 densitet av fluid
        d: float   tjocklek av superfilmen
        d0: float  tjocklek av ett lager av superfluid, atomlager
        M4: float  molmassa för helium 4
        M3: float  molmassa för helium 3
    """
    r: float # radien på öppningen vid stillen. brukar vara runt 1 mm 

    tabeller: TermiskData = field(default_factory=lambda: standardtabeller)

    rho_sup: float = 145  #kg/m^3 densitet av fluid
    d: float = 30e-9 #tjocklek av superfilmen
    d0: float = 0.36e-9 #tjocklek av ett lager av superfluid, atomlager
    M4: float = M_He4 #molmassa för helium 4
    M3: float = M_He3 #molmassa för helium 3

    
    def _mu_He4(self, T: float, X: float) -> float:
        """
        Kemisk potential för He-4 i den utspädda He-3/He-4-lösningen [J/mol].

        Värdena baseras på tabelldata från Radebaugh (1967), tabell 10

        Returvärdet motsvarar den molära kemiska potentialen för He-4
        vid given temperatur och koncentration.

        """
        
        return self.tabeller.kemisk_potential4(T, X) 

    def _lös_jämvikt(
        self,
        T_mc: float,
        X_mc: float,
        X_v_mål: float,
        startvärde: tuple[float, float] = (1, 0.001),
    ) -> tuple[float, float]:
        """
        Löser jämviktsvillkoren för stillkammaren.

        Hittar T_s och X_s så att:
          1. μ_4(T_s, X_s) = μ_4(T_mc, X_mc)
          2. X_v(T_s, X_s) = X_v_mål

        Raises RuntimeError om lösaren inte konvergerar.
        """
        mu_4_mc = self._mu_He4(T_mc, X_mc)

        def ekvationer(v):
            T_s, X_s = v
            return [
                (self._mu_He4(T_s, X_s) - mu_4_mc) / abs(mu_4_mc),
                self.tabeller.Xv(T_s, X_s) - X_v_mål,
            ]

        lösning = least_squares(
            ekvationer,
            startvärde,
            bounds=([0.01, 1e-6], [2.0, 0.064]), # körde [2.0, 0.999] förut
        )

        if lösning.cost > 1e-6:
            raise RuntimeError(
        f"Stillkammare.lös_jämvikt konvergerade inte, kostnad={lösning.cost:.2e}"
    )

        return float(lösning.x[0]), float(lösning.x[1])

    def egenskaper(
        self,
        T_mc: float,
        X_mc: float,
        X_v_mål: float,
    ) -> tuple[float, float, float, float]:
        """
        Returnerar (T_s, X_s, X_v, p_v) för stillkammaren.
        """
        T_s, X_s = self._lös_jämvikt(T_mc, X_mc, X_v_mål)
        X_v = float(np.clip(self.tabeller.Xv(T_s, X_s), 0.0, 1.0))
        p_v = max(0.0, float(self.tabeller.pv(T_s, X_s)))*133.322 #konverterar till Pa
        return T_s, X_s, X_v, p_v
    


    def molflöde(
        self,
        T_s: float,    # Stilltemperatur [K]
        X_s: float,    # He-3 molfraktion i vätskan
        P_v: float,    # Ångtryck [Pa]
        X_v: float,    # Molfraktion i gasen
    ) -> tuple[float, float, float]:
        """
        Beräknar totalt molflöde ut ur stillkammaren [mol/s].
        Summan av:
        n1 - krypflöde via superfilm längs rörväggen
        n2 - gasfasflöde (soniskt/subsoniskt) genom röret
        
        Parametrar
        ----------
        T_s  : Stilltemperatur [K]
        X_s  : He-3 molbråk i vätskan vid stillen
        P_v  : Ångtryck [Pa] (från egenskaper())
        """


        # --- Rörgeometri ---
        A = np.pi * self.r ** 2   # Tvärsnittsarea [m²]


        # --- Superfilmskrypning (He-4 krypflöde längs rörväggen) ---
        # Critical superfluid speed
        ny_v = 1.23e-8 / self.d 

        #Molflöde från från krypning
        n1 = 2*self.r*np.pi*self.rho_sup*ny_v*(self.d-self.d0)/self.M4


        # --- Gasfasflöde genom röret ---
        # genomsnittliga olmassan
        M_s = X_s*self.M3+(1-X_s)*self.M4
        # desnisteten på gasen i stillen
        rho_g = P_v*M_s/(R*T_s)
        #Ljudhastigheten i gasen
        cs1 = np.sqrt(gamma*R*T_s/M_s)

        #Critical pressure 
        Pc=(2/(gamma+1))**(gamma/(gamma-1))*P_v


        # --- Gas molflöde (He-3 från destillatorn) ---
        #Term uppdelning
        konst = (2/(gamma-1))**(1/2)
        term1 = (Pc/P_v)**(1/gamma)
        term2 = (1 - (Pc/P_v)**((gamma-1)/gamma))**(1/2)

        #Molflöde från gasen i destillatorn
        n2 =  (A * rho_g * cs1 * konst * term1 * term2) / M_s 

        # --- Totalt flöde ut ur stillen ---
        m_prick = n1 + n2

        X_in = (X_v*n2)/(m_prick)

        return m_prick, Pc, X_in

# ------------------------------------------------------------------
# Pumpen
# ------------------------------------------------------------------

@dataclass
class Pump:
    """
    Modell för cirkulationspumpen.

    Modelleras som en ideal vakuumpump och beräknar den effektiva pumphastigheten för specefikt molflöde.
    Gasen behandlas som idealgas.
    """

    def s_pump_critical(self, T_rum: float, p_inlopp: float, m_prick: float) -> float:
        """
        Molflöde som pumpas från stillen.

        Härleds från idealgaslagen tillämpat på pumpens inlopp:

            ṅ = S · (p_inlopp) / (R · T_s)

        Parametrar
        ----------
        T_rum    : float   rumstemperatur vid pumpen  [K] 
        p_inlopp : float   Tryck vid pumpens inlopp   [Pa]
        m_prick  : float   Molflödet från stillen     [mol/s]

        Returvärde
        ----------
        float
            S_min : float  Effektiv pumphastighet     [m^3/s]
        """

        return (m_prick * R * T_rum )/(p_inlopp)
    

# ------------------------------------------------------------------
# Värmeväxlaren  
# ------------------------------------------------------------------
    
@dataclass
class Värmeväxlare:
    """
    Motströmsvärmeväxlare mellan strömmarna mot och ifrån blandningskammaren.
 
    Löser randvärdesproblemen (BVP) från Pradhan et al. (2013), ekv. 13-14,
    baserade på Frossati/Takano-modellen för kontinuerlig motströmsväxlare.
 
    ODE-systemet i den koordinat x som löper från 0 (varmt ände, stillen)
    till L (kallt ände, blandningskammaren):
 
        T_c · dT_c/dx = (0.01·p / (ṁ·R_kt)) · (T_d⁴ - T_c⁴)      (ekv. 14)
        T_c · dT_c/dx = (C_c/C_d) · T_d · dT_d/dx                  (ekv. 13)
 
    Randvillkor (motström):
        T_c(x=0) = T_varm_in   — koncentrerad ström in från stillen
        T_d(x=L) = T_kall_in   — utspädd ström in från blandningskammaren
 
    Returvärden från utloppstemperaturer():
        T_till_BK   = T_c(x=L)  — koncentrerad ström ut, in i blandningskammaren
        T_till_still = T_d(x=0) — utspädd ström ut, in i stillen
 
    Parametrar
    ----------
    L     : float   Värmeväxlarens längd  [m]
    R_kt  : float   Kapitza-resistivitetskoefficient  [K⁴·m²/W]
    p     : float   Effektiv värmeöverföringsyta per längdenhet  [m²/m]
    C_c   : float   Specifik värmekapacitet, koncentrerad sida  [J/(mol·K²)]
                    Standardvärde 22 J/(mol·K²) från Pradhan ekv. 7
    C_d   : float   Specifik värmekapacitet, utspädd sida  [J/(mol·K²)]
                    Standardvärde 106 J/(mol·K²) från Pradhan ekv. 7
    """
 
    L:    float         # värmeväxlarens längd  [m]
    p:    float         # ytarea per längdenhet  [m²/m]


    tabeller: TermiskData = field(default_factory=lambda: standardtabeller)

    C_c:  float = 22.0  # spec. värme koncentrerad sida  [J/(mol·K²)]
    C_d:  float = 106.0 # spec. värme utspädd sida  [J/(mol·K²)]

    L_d: float = 1.0         # Längd i meter
    L_c: float = 5.0         # Längd i meter
    N: int = int(1200)       # Mängd sektioner, fler sektioner gör uträkningen långsammare, men ger slätare resultat

    def __post_init__(self):
        # Beräknade attribut 
        self.Z_d    = 8 * self.L_d / (np.pi * r_vvx**4)     # Impedans enligt Hagen-Pausille
        self.Z_c    = 8 * self.L_c / (np.pi * r_vvx**4)     # Impedans enligt Hagen-Pausille
        self.dL_d   = self.L_d / self.N                     # Längd per stycke
        self.dL_c   = self.L_c / self.N                     # Längd per stycke
        self.L_pos_d = np.linspace(0, self.L_d, self.N)     # Segment i L_d
        self.L_pos_c = np.linspace(0, self.L_c, self.N)     # Segment i L_c

    def _PiV_get(self, T: float, X: float) -> float:
        """
        Korrektionsterm ΠV för He-4 i lösningen [J/mol].

        Termen representerar avvikelsen i kemisk potential mellan
        rent He-4 och He-4 i He-3/He-4-lösningen.

        Parametrar
        ----------
        T : float
            Temperatur [K]

        X : float
            He-3-molfraktion i lösningen [-]

        Returvärde
        ----------
        U4: float
            ΠV-term [J/mol]

        Källa
        -----
        Radebaugh (1967), tabell 10.
        """
        return self.tabeller.kemisk_potential4(T, X) 
    
    def _mu_40_get(self, T: float) -> float:
        """
        Kemisk potential för rent He-4 [J/mol].

        Parametrar
        ----------
        T : float
            Temperatur [K]

        Returvärde
        ----------
        float
            Kemisk potential för rent He-4 [J/mol]
        Källa
        -----
        Radebaugh (1967), tabell 5.
        """
        return self.tabeller.mu40_He4(T) 
    

    def _mu_4_get(self, T, x):
        """
        Kemisk potential för He-4 i lösningen [J/mol].

        Beräknas som kemisk potential för rent He-4 minus
        korrektionstermen ΠV:

            μ₄ = μ₄⁰ - ΠV

        Parametrar
        ----------
        T : float
            Temperatur [K]

        X : float
            He-3-molfraktion i lösningen [-]

        Returvärde
        ----------
        float
            Kemisk potential för He-4 i lösningen [J/mol]

           Källa
        -----
        Pradhan et al. Källa 1, ekvation 2.
        """
        mu_4 = self._mu_40_get(T) - self._PiV_get(T, x)  
        return mu_4  


    def _residual(self, x, T_matris, mu_4_mc, i):
        """
        Residual för brentq-lösningen av He-3-molfraktionen på utspädd sida.

        Söker det x (molfraktion He-3) som ger kemisk jämvikt med
        blandningskammaren, dvs. μ₄(T_d, x) = μ₄_MC.

        Parametrar
        ----------
        x        : float        Gissad He-3-molfraktion på utspädd sida [-]
        T_matris : np.ndarray   Temperaturmatris, rad 0 = konc., rad 1 = utspädd
        mu_4_mc  : float        Kemisk potential för He-4 i blandningskammaren [J/mol]
        i        : int          Aktuellt diskretiseringssteg

        Returvärde
        ----------
        float   μ₄(T_d[i], x) - μ₄_MC  (= 0 vid jämvikt)

        """
        return self._mu_4_get(T_matris[1, i], x) - mu_4_mc


    # Funktioner för värmeöverföring

    def _V_get(self, x):
        """Molvolym för He-3/He-4-lösningen [m³/mol]."""
        V_cm3 = (27.58 - 3.3 * x ** 3)  # Källa 1 ekv 3
        V = V_cm3 * 1e-6  # Notera cm3 till m3 factorn
        return V  # [m^3/mol]


    def _Qvisc_get(self, T, x, Z, sida, m_prick):

        """ 1) Volymflöde (m^3/s) """
        V_punkt = self._V_get(x) * m_prick

        """ 2) Viskositet (Pa*s)"""
        if sida == 1:  # logik som säger om du räknar på den koncentrerade sidan eller den utspädda sidan
            eta = 1.81e-6 / T**2  # conc
        else:
            eta = 510e-7 / T**2  # dilute

        """ 3) dP/dx för laminärt flöde i rör enligt Hagen-Poiseuille: dP/dx = (8 * eta * V) / (pi * r^4)"""
        dPdL = Z * eta * V_punkt   # Pa/m

        """ 4) Viskös uppvärmning (effekt) i sektionen: Qvisc = ΔP * Q"""
        Qvisc = dPdL * V_punkt                          # W/m

        return Qvisc


    def _R_kt_get(self, T_c, T_d):
        """
        Total Kapitza-resistans för gränsskiktet mellan koncentrerad
        och utspädd sida [m²·K/W].

        Ekv. 2a-2b i källan.

        """
        if T_c < 0.13:
            # Uträknat i SI-Enheter [m^2*K/W]
            rho_c = (20e-3)/(T_c**3)
        else:
            # Uträknat i SI-Enheter [m^2*K/W]
            rho_c = (2.4/(T_c**4)+1.55/(T_c**3))*1e-3
        # Uträknat i SI-Enheter [m^2*K/W]
        rho_d = 7e-3/(T_d**3)
        return rho_c+rho_d


    def _dTcdx_radial_get(self, T_c, T_d, m_prick):
        """
        Radiellt bidrag till temperaturgradient på koncentrerad sida [K/m].
        """
        dTcdx_radial = ((1/88)*self.p/T_c)*(T_d**4 - T_c**4) / (m_prick*self._R_kt_get(T_c, T_d))  # ekvation 13 källa 1
        return dTcdx_radial


    def _dTddx_radial_get(self, T_c, T_d, m_prick):
        """
        Radiellt bidrag till temperaturgradient på utspädd sida [K/m].
        """

        dTddx_radial = T_c/((106/22)*T_d)*self._dTcdx_radial_get(T_c, T_d, m_prick)  # ekvation 14 källa 1
        return dTddx_radial


    def _dTcdx_get(self, T_c, T_d, x, m_prick):
        """
        Total temperaturgradient på koncentrerad sida [K/m].

        Summan av radiellt värmeöverföringsbidrag och viskös uppvärmning:
            dT_c/dx = dT_c/dx|_radiell + Q_visc / (ṁ · C_c · T_c)
        
        """
        
        dTcdx_visc = self._Qvisc_get(T_c, x, self.Z_c, 1, m_prick)/(m_prick*22*T_c)
        dTcdx = self._dTcdx_radial_get(T_c, T_d, m_prick) + dTcdx_visc
        return dTcdx


    def _dTddx_get(self, T_c, T_d, x, m_prick):
        """
        Total temperaturgradient på utspädd sida [K/m].

        Summan av radiellt värmeöverföringsbidrag och viskös uppvärmning:
            dT_d/dx = dT_d/dx|_radiell + Q_visc / (ṁ · C_c · T_c)
        """

        dTddx_visc = self._Qvisc_get(T_d, x, self.Z_d, 0, m_prick)/(m_prick*22*T_c)
        dTddx = self._dTddx_radial_get(T_c, T_d, m_prick) + dTddx_visc
        return dTddx


    def _VVX_Loop(
            self,
            x : float,
            T_mc :float,
            x_mc : float,
            m_prick : float,
            x_in : float,
            mc : float,
            plotta : int
            )-> float:
        """
        Integrerar temperaturprofilen längs värmeväxlaren från blandnings-
        kammarsidan (i = N-1) mot stillen (i = 0).

        Randvillkor:
            T_c(N-1) = 0.7 K  (kondensationstemperatur)
            T_d(N-1) = x      (gissad inloppstemperatur utspädd sida)

        I varje steg bestäms He-3-molfraktionen x_d via kemisk jämvikt
        (brentq), varefter temperaturerna uppdateras med Euler-steget.

        Parametrar
        ----------
        x       : float   Gissad temperatur T_d vid blandningskammaren [K]
                          (kallas 'x' eftersom funktionen används i brentq)
        T_mc    : float   Temperatur i blandningskammaren [K]
        x_mc    : float   He-3-molfraktion i blandningskammaren [-]
        m_prick : float   Molflöde [mol/s]
        x_in    : float   He-3-molfraktion på koncentrerad sida (konstant) [-]
        mc      : int     Returläge: 1 → returnera T_c(0), 0 → returnera residual
        plotta  : int     1 → rita temperatur- och molfraktionsprofiler

        Returvärde
        ----------
        float
            mc=1: T_c vid stillen [K]
            mc=0: T_d(0) - T_mc  (residual för yttre brentq)
        """

        T_guess = x  # namnbyte för tydlighet (x används av brentq)
        mu_4_mc = self._mu_4_get(T_mc, x_mc)

        # Matris: rad 0 = koncentrerad sida, rad 1 = utspädd sida
        # Kolonnindex löper från blandningskammaren (N-1) mot stillen (0)
        T_matris = np.zeros((2, self.N))
        x_matris = np.zeros((2, self.N))  

        # Startvillkor vid blandningskammaren
        T_matris[0, self.N-1] = 0.7  # T_c: kondensationstemperatur [K]
        T_matris[1, self.N-1] = T_guess  # T_d: gissad inloppstemperatur [K]
        x_matris[0, :] = x_in  # molfraktion konc. sida är konstant


        for i in range(self.N-1, 0, -1):
            # Bestäm He-3-molfraktion på utspädd sida via kemisk jämvikt
            x_matris[1, i] = sp.brentq(
                self._residual, 0.0, 0.12, xtol=0.001, args=(T_matris, mu_4_mc, i)
                )
            
            T_matris[0, i-1] = T_matris[0, i] + self._dTcdx_get(T_matris[0, i], T_matris[1, i],
                                                                x_matris[0, i], m_prick)*self.dL_c  # DeltaT per steg  
                      
            T_matris[1, i-1] = T_matris[1, i] + self._dTddx_get(T_matris[0, i], T_matris[1, i], 
                                                                x_matris[1, i], m_prick)*self.dL_d


        if plotta == 1:
            # ── Temperaturprofil (båda sidor) ────────────────────────────

            step = 25
            fig, ax1 = plt.subplots()

            # Skapa index och säkerställ att sista punkten inkluderas
            idx = list(range(0, len(self.L_pos_c), step))

            if idx[-1] != len(self.L_pos_c) - 1:
                idx.append(len(self.L_pos_c) - 1)

            idx = np.array(idx)

            L_pos_blue_scaled = self.L_pos_c[idx] / 5

            # Primär x-axel (blå, 0-1 m) - nederst
            ax1.plot(
                L_pos_blue_scaled,
                T_matris[1, idx],
                label='T_d (utspädd fas)',
                color='blue',
                marker='o'
            )

            ax1.set_xlabel("Axiell position L_d [m] (utspädd fas)", color='blue')
            ax1.set_xlim(0, 1)
            ax1.tick_params(axis='x', labelcolor='blue')

            # Sekundär x-axel (röd, 0-5 m) - överst
            ax2 = ax1.twiny()

            ax2.plot(
                self.L_pos_c[idx],
                T_matris[0, idx],
                label='T_c (koncentrerad fas)',
                color='red',
                marker='s'
            )

            ax2.set_xlabel("Axiell position L_c [m] (koncentrerad fas)", color='red')
            ax2.set_xlim(0, 5)
            ax2.tick_params(axis='x', labelcolor='red')

            ax1.set_ylabel("Temperatur [K]")

            plt.title("Temperaturprofil av värmeväxlare i koncentrerad och utspädd fas")

            # Kombinera legends från båda axlarna
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            ax2.legend(lines1 + lines2, labels1 + labels2)

            plt.show()

            # ── Molfraktion utspädd sida ──────────────────────────────────
            plt.figure()
            plt.plot(self.L_pos_d[0:self.N-1:step], x_matris[1, 1::step], color='blue', marker='o')
            plt.xlabel("Axiell position L_d (utspädd fas) [m]")
            plt.ylabel("Molfractionen av utspädd fas (X_d)")
            plt.title("Temperaturprofil av värmeväxlare i utspädd fas")
            plt.show()

            # ── Molfraktion koncentrerad sida ─────────────────────────────
            plt.figure()
            plt.plot(self.L_pos_c[::35], x_matris[0, ::35], color='red', marker='s')
            plt.xlabel("Axiell position L_c (koncentrerad fas) [m]")
            plt.ylabel("Molfractionen av koncentrerad fas (X_c)")
            plt.title("Temperaturprofil av värmeväxlare i koncentrerad fas")
            plt.show()


        if mc == 1:
            return T_matris[0,0]            # T_c vid stillen
        else:
            return T_matris[1, 0] - T_mc    # residual: T_d(still) − T_MC
        

    def brent_VVX(
        self,
        T_mc: float,
        x_mc: float,
        m_prick: float,
        x_in: float,
        plotta: int,
    ) -> tuple[float, float]:
        """
        Löser randvärdesproblemen för värmeväxlaren med brentq

        Söker den ingångstemperatur T_d vid blandningskammaren som gör att
        T_d vid stillen stämmer överens med T_mc (randvillkoret).

         Parametrar
        ----------
        T_mc    : float   Temperatur i blandningskammaren [K]
        x_mc    : float   He-3-molfraktion i blandningskammaren [-]
        m_prick : float   Molflöde [mol/s]
        x_in    : float   He-3-molfraktion på koncentrerad sida [-]
        plotta  : int     1 → generera temperatur- och molfraktionsplotter

        Returvärde
        ----------
        tuple[float, float]
            (T_till_MC, T_till_still)
            T_till_MC    : temperatur koncentrerad sida in i blandningskammaren [K]
            T_till_still : temperatur utspädd sida in i stillen [K]
        """
        # Steg 1: hitta T_d vid blandningskammaren (yttre randvillkor)
        T_in_i_still = sp.brentq(self._VVX_Loop, T_mc+0.001, 0.7, xtol=0.00001, args=(T_mc, x_mc, m_prick, x_in, 0, 0))

        # Steg 2: beräkna T_c vid stillen med det funna randvillkoret
        T_in_i_MC = self._VVX_Loop(T_in_i_still, T_mc, x_mc, m_prick, x_in, 1, plotta)
        return T_in_i_MC, T_in_i_still


# ------------------------------------------------------------------
# Fluid  
# ------------------------------------------------------------------

@dataclass
class Fluid:
    """
    Modellerar He-3-gasflöde i ett rör med varierande temperatur och tryck.

    Hanterar laminärt, turbulent och molekylärt flöde (Knudsen-regim)
    samt övergångsregimer. Beräknar utgångstrycket P_d givet ett känt
    flöde m_prick och inloppstryck P_u.

    Parametrar
    ----------
    L   : float   Rörlängd [m]
    r   : float   Rörradie [m]
    n   : float   Antal böjningar i röret [-]
    T_1 : float   Temperatur vid rörstart [K]
    T_2 : float   Temperatur vid rörsslut [K]
    """

    L:   float   # Längden på röret
    r:   float   # Radien på röret
    n:   float   # Antal böjningar i röret
    T_1: float   # Temperaturen i början av röret
    T_2: float   # Temperaturen i slutet av röret

    epsilon: float = 1e-6   # Rörytans råhet [m] — saknades tidigare

    S_f: float = 1.0   # Shape factor
    S:   float = 0.4   # Pumping speed
    S_u: float = 0.4  # Upstream pumping speed (sätts i __post_init__ om S_u = S)

   
    # ── Hjälpfunktioner ───────────────────────────────────────────────


    def _f_d_get(self, Re, epsilon, D_h):
        """ Darcy-Weisbach friktionsfaktor [-]."""
        Re_star = Re / self.S_f
        if Re_star <= 2300:
            return 64 / (Re_star + 0.01)
        elif Re_star >= 3500:
            return 1.6364 / (
                (np.log((6.9 / Re_star) + ((self.S_f * epsilon) / (3.7 * D_h)) ** 1.11)) ** 2
            )
        else:
            fd_lam  = 64 / Re_star
            fd_turb = 1.6364 / (
                (np.log((6.9 / Re_star) + ((self.S_f * epsilon) / (3.7 * D_h)) ** 1.11)) ** 2
            )
            return np.interp(Re_star, [2300, 3500], [fd_lam, fd_turb])

    def _h_f_get(self, f_d, L_c, D_h, n_c):
        """ Dimensionslös friktionsförlust (head loss) [-]"""
        return n_c + (f_d * L_c) / D_h

    def _Ma_x_get_unchoked(self, K, Ma_n):
        """ Machs tal vid utloppet för icke-chokat flöde [-]"""
        Ma_x = np.sqrt(
            (np.sqrt(1 + (2 * (gamma - 1) * (K * Ma_n) ** 2) /
             (1 + ((gamma - 1) * Ma_n ** 2) / 2) ** ((gamma + 1) / (gamma - 1))) - 1)
            / (gamma - 1)
        )
        return Ma_x

    def _Ma_n_unchoked_func(self, x, K, h_f):
        """Residualfunktion för bestämning av Ma_n vid icke-chokat flöde"""

        Ma_n  = x
        Ma_x  = self._Ma_x_get_unchoked(K, Ma_n)
        return (
            (1 / gamma) * (
                (1 / Ma_n ** 2) - (1 / Ma_x ** 2)
                - ((gamma + 1) / 2) * np.log(
                    (Ma_x / Ma_n) ** 2
                    * ((gamma - 1) * Ma_n ** 2 + 2)
                    / ((gamma - 1) * Ma_x ** 2 + 2)
                )
            ) - h_f
        )

    def _Ma_n_get_numer_unchoked(self, K, h_f):
        """Löser Ma_n för icke-chokat flöde numeriskt med brentq"""
        return spo.brentq(
            self._Ma_n_unchoked_func, 1e-5, 1, xtol=1e-12, args=(K, h_f)
        )

    def _T_0_n_get(self, K, h_f):
        """ Normaliserad stagnationstemperatur vid inloppet [-]"""
        Ma_n = self._Ma_n_get_numer_unchoked(K, h_f)
        return 1 + Ma_n ** 2 * (gamma - 1) / 2

    def _T_0_x_get(self, K, h_f):
        """ Normaliserad stagnationstemperatur vid utloppet [-]"""
        Ma_n = self._Ma_n_get_numer_unchoked(K, h_f)
        Ma_x = self._Ma_x_get_unchoked(K, Ma_n)
        return 1 + Ma_x ** 2 * (gamma - 1) / 2

    def _C_z_get(self, T_0, r):
        """ Karakteristisk konduktans för ljudhastighetsbegränsat flöde [m³/s]"""
        return np.pi * r ** 2 * np.sqrt(R * T_0 / M_He3)

    def _viscosity_He3_gas(self, T_0):
        """He-3 viskositet, kinetisk teori behöver läsa på mer"""
        return 2 * np.sqrt(m_He3 * k_b * T_0) / (sigma * 3 * np.sqrt(np.pi))

    def _poiseuille_conductance(self, T_0, L, r):
        """Dynamisk viskositet för He-3-gas [Pa·s] (kinetisk teori)"""
        eta = self._viscosity_He3_gas(T_0)
        return np.pi * r ** 4 / (8 * eta * L)

    def _Q_v_Low_Flow_get(self, Re, f_d, P_d, K, T_0_d, T_0_u, L_c, D_h, n_c, L, r):
        """ Flöde i lågflödesregimen [m³/s]"""
        C_d  = self._poiseuille_conductance(T_0_d, L, r)
        C_z  = self._C_z_get(T_0_u, r)
        Re_star = Re / self.S_f
        if Re_star < 2300:
            return ((C_z * P_d) / (1 + n_c)) * (
                np.sqrt((C_z / C_d) ** 2 + (n_c + 1) * (K ** 2 - 1)) - C_z / C_d
            )
        else:
            return C_z * P_d * np.sqrt((K ** 2 - 1) / (n_c + 1 + (f_d * L_c / D_h)))

    def _Ma_n_choked_func(self, x, h_f):
        """ Residualfunktion för bestämning av Ma_n vid chokat (Ma_x = 1) flöde"""
        Ma_n = x
        return (
            (1 / gamma) * (
                (1 / Ma_n ** 2) - 1
                - ((gamma + 1) / 2) * np.log(
                    (1 / Ma_n) ** 2
                    * ((gamma - 1) * Ma_n ** 2 + 2)
                    / ((gamma - 1) * 1 ** 2 + 2)
                )
            ) - h_f
        )

    def _Ma_n_get_numer_choked(self, h_f):
        """ Löser Ma_n för chokat flöde numeriskt med brentq"""
        return spo.brentq(
            self._Ma_n_choked_func, 1e-7, 1, xtol=1e-12, args=(h_f,)
        )

    def _K_c_get(self, Ma_n):
        """ Kritiskt tryckförhållande K_c vid chokat flöde (Ma_x = 1) [-]"""
        return (
            np.sqrt((gamma + 1) / 2)
            * (1 + ((gamma - 1) * Ma_n ** 2) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
            / Ma_n
        )

    def _Q_v_choked_get(self, K_c, P_u, T_0_u, r):
        """ Volymflöde vid chokat (ljudhastighetsbegränsat) flöde [m³/s]"""
        return np.sqrt((gamma * (gamma + 1)) / 2) * (self._C_z_get(T_0_u, r) * P_u / K_c)

    def _Q_v_unchoked_get(self, T_0_u, Ma_n, P_u, r):
        """Volymflöde vid icke-chokat (subsoniskt) flöde [m³/s]."""
        return (
            np.sqrt(gamma) * self._C_z_get(T_0_u, r) * P_u * Ma_n
            / (1 + 0.5 * (gamma - 1) * Ma_n ** 2) ** ((gamma + 1) / (2 * (gamma - 1)))
        )

    def _C_m_cylindrical_get(self, T_0, L, r):
        """ Molekylär konduktans för cylindrisk rörgeometri [m³/s]"""
        l_e   = L * (1 + 1 / (3 + (3 * L) / (7 * r)))
        alpha = 1 / (1 + (3 * l_e) / (8 * r))
        C_a   = self._C_z_get(T_0, r) / np.sqrt(2 * np.pi)
        return C_a * alpha

    def _Q_m_get(self, P_u, P_d, T_0_u, L, r):
        """ Molekylärt (Knudsen) flöde [m³/s]"""
        return self._C_m_cylindrical_get(T_0_u, L, r) * (P_u - P_d)

    def _mean_free_path(self, T, P):
        """Medelfri väglängd för He-3 [m]"""
        return k_b * T / (np.sqrt(2) * np.pi * d_mol ** 2 * P)

    def _Kn_get(self, lambd, D_h):
        """ Knudsentalet Kn = λ/D_h [-]"""
        return lambd / D_h

    def _phi_Kn(self, Kn, L, D_h):
        """Knudsen number function för övergångsregim (Livesey stycke 2)"""
        phi_s = 3 * np.pi / (128 * Kn)
        phi_l = 3 * np.pi * (1 + 4 * Kn) / (128 * (1 + Kn) * Kn ** 0.55)
        return phi_s + (phi_l - phi_s) * L / (L + D_h)

    def _Q_get(self, Q_v, Q_m, Phi):
        """ Totalt flöde som summa av viskös och molekylär del [m³/s]"""

        return Q_v + Q_m / (1 + Phi)

    def _eta_sat_He3(self, T):
        """Huang et al., Cryogenics 52 (2012) 538-543, Eq. (1)"""
        c1 =  2.897e-7
        c2 = -7.02e-7
        c3 =  2.012e-6
        c4 =  1.323e-6
        return c1 / T ** 2 + c2 / T ** 1.5 + c3 / T + c4

    def _Re_get(self, Q, T_0_u, r):
        """ Reynoldstal för He-3-flöde i cirkulärt rör [-]"""
        return 4 * M_He3 * Q / (R * T_0_u * 2 * np.pi * r * self._eta_sat_He3(T_0_u))

    def _Visc_Ma_n_get(self, f_d, T_0_u, C_u, n_c, L_c, D_h):
        """ Beräknar Mach-talet Ma_n vid inloppet med hänsyn till viskös friktion [-]"""
        h_f  = self._h_f_get(f_d, L_c, D_h, n_c)
        Ma_n = sympy.Symbol('Ma_n')
        C_z  = self._C_z_get(T_0_u, self.r)
        sympy.solve(
            n_c + (2 / ((gamma ** 2) * Ma_n)) * (C_z / C_u)
            * (1 + ((gamma - 1) * Ma_n ** 2) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
            - h_f,
            Ma_n,
        )
        return Ma_n

    def _brentsolve(
        self,
        x: float,
        BV: int,
        P_u: float,
        Q_faktisk: float,
        T_u: float,
        T_d: float,
        L: float,
        r: float,
        n_c: float,
        delta: float,
    ) -> float:
        """
        Residualfunktion för brentq-sökning av nedströmstrycket P_d.

        Itererar internt tills Reynoldstalet konvergerat (|Re_guess - Re_utr| ≤ 10),
        och returnerar sedan Q_beräknat - Q_faktisk.

        Flödesregimen bestäms av kriteriet BV:
            BV=1: pumphastighetskriterium (S / C_z < 0.1)
            BV=2: uppströms pumphastighetskriterium
            BV=3: tryckverhållande vs. Knudsen-längd

        Parametrar
        ----------
        x         : float   Gissad P_d [Pa]
        BV        : int     Kriteriumval för lågflödesregim (1, 2 eller 3)
        P_u       : float   Uppströmstryck [Pa]
        Q_faktisk : float   Genomflöde [Watt]
        T_u       : float   Uppströms temperatur [K]
        T_d       : float   Nedströms temperatur [K]
        L         : float   Rörlängd [m]
        r         : float   Rörradie [m]
        n_c       : float   Antal böjningar [-]
        delta     : float   Geometrisk faktor  [-]

        Returvärde
        ----------
        float   Q_beräknat - Q_faktisk  (= 0 vid korrekt P_d)
        """

        P_d_guess = x
        K   = P_u / P_d_guess
        D_h = 2 * r
        L_c = L + 0.41 * D_h # effektiv rörlängd med inloppskorrigering
       
        # Iterera tills Reynoldstalet konvergerat
        Q_utr  = 0
        Re_utr = 2300 * self.S_f
        Re_guess = 0

        while np.abs(Re_guess - Re_utr) > 10:
            
            # Beräkna friktionsfaktor, friktionsförlust och hjälpstorheterna
            Re_guess = Re_utr
            f_d  = self._f_d_get(Re_guess, self.epsilon, D_h)
            h_f  = self._h_f_get(f_d, L_c, D_h, n_c)
            C_u  = self._poiseuille_conductance(T_u, L, r)
            lambd = self._mean_free_path(T_u, P_u)
            Kn   = self._Kn_get(lambd, D_h)
            Phi  = self._phi_Kn(Kn, L, D_h)
            
            # ── Lågflödeskriterier (BV = boundary value type) ────────────
            criterion_1 = self.S  / self._C_z_get(T_d, r)
            criterion_2 = self.S_u * (
                np.sqrt((self._C_z_get(T_u, r) / C_u) ** 2 + n_c + 1 / (delta ** 2))
                - self._C_z_get(T_u, r) / C_u
            )
            criterion_3 = D_h * np.sqrt(np.pi / 2) * (K - 1) / (3.2 * self.S_f * Kn)

            low_flow = (
                (BV == 1 and criterion_1 < 0.1)
                or (BV == 2 and criterion_2 < self._C_z_get(T_u, r))
                or (BV == 3 and criterion_3 < L)
            )

            if low_flow:
                #Lågflödesregim
                Q_v = self._Q_v_Low_Flow_get(Re_guess, f_d, P_d_guess, K, T_d, T_u, L_c, D_h, n_c, L, r)
            else:
                Ma_n_choked = self._Ma_n_get_numer_choked(h_f)
                K_c = self._K_c_get(Ma_n_choked)
                if K_c <= K:
                    Q_v = self._Q_v_choked_get(K_c, P_u, T_u, r)
                else:
                    try:  
                        Ma_n_unchoked = self._Ma_n_get_numer_unchoked(K, h_f)
                        Q_v = self._Q_v_unchoked_get(T_u, Ma_n_unchoked, P_u, r)
                    except ValueError:
                        Q_v = self._Q_v_Low_Flow_get(Re_guess, f_d, P_d_guess, K, T_d, T_u, L_c, D_h, n_c, L, r)
            
            # Totalt flöde med molekylärt bidrag
            Q_m   = self._Q_m_get(P_u, P_d_guess, T_u, L, r)
            Q_utr = self._Q_get(Q_v, Q_m, Phi)
                        
            # Uppdatera Reynoldstalet för nästa iteration
            Re_utr = self._Re_get(Q_utr, T_u, r)

        return Q_utr - Q_faktisk

    def P_get_BV_2(
        self,
        BV: int,
        P_u: float,
        Q_faktisk: float,
        T_0_u: float,
        T_0_d: float,
        L: float,
        r: float,
        n_c: float,
        delta: float,
    ) -> float:
        """  Beräknar nedströmstrycket P_d givet ett känt flöde Q_faktisk och uppströmstrycket [Pa]"""
        P_d = spo.brentq(
            self._brentsolve, 1e-1, P_u*0.99999, xtol=1e-3,
            args=(BV, P_u, Q_faktisk, T_0_u, T_0_d, L, r, n_c, delta),
        )
        return P_d