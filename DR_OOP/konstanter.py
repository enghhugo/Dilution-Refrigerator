"""
konstanter.py
=============
Fysikaliska konstanter och systemparametrar för simulering av en
utspädningskylare (dilution refrigerator).

Modulen innehåller:
- fundamentala fysikaliska konstanter,
- temperatur- och tryckrelaterade referensvärden,
- samt systemspecifika parametrar för kylarsimuleringen.

Alla SI-enheter används om inget annat anges.

"""

from __future__ import annotations
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------
# Sökvägar
# ------------------------------------------------------------------

#: Projektets rotkatalog (en nivå ovanför DR_OOP/)
PROJEKT_ROOT = Path(__file__).parent.parent

#: Mapp som innehåller alla Excel-datafiler
DATA_MAPP = PROJEKT_ROOT / "data"

# ------------------------------------------------------------------
# Fysikaliska konstanter  (SI-enheter genomgående)
# ------------------------------------------------------------------

#: Universella gaskonstanten  [J / (mol·K)]
R: float = 8.314_462_618

#: Plancks konstant  [J·s]
h: float = 6.626_070_15e-34

#: Boltzmanns konstant  [J/K]
k_b: float = 1.380_649e-23

#: Avogadros tal  [1/mol]
N_A: float = 6.022_140_76e23

#: idealt förhållande för helium termodynamiskt
gamma: float = 5/3 

# ------------------------------------------------------------------
# He-3 / He-4 blandningskonstanter
# (Radebaugh 1967, Tabell 3 och omgivande text)
# ------------------------------------------------------------------


#: Sommerfeldkoefficient, utspädd fas    [J / (mol·K²)]
gamma_d: float = 95.0

#: Sommerfeldkoefficient, koncentrerad fas  [J / (mol·K²)]
gamma_c: float = 12.0

#: Molvolym för rent He-3 vid 0 K  [m³/mol]  (~27.58 cm³/mol)
v_m: float = 27.58e-6

#molmassa för helium 4 [kg/mol]
M_He4: float = 4.0026e-3

# molmassa för helium 3 [kg/mol]
M_He3: float = 3.016e-3            

# massa per He-3-atom [kg]
m_He3 = M_He3 / N_A         

# molekylär diameter He-3 [m]
d_mol = 2.6e-10            

sigma = np.pi * d_mol**2    # molekylär tvärsnittsarea [m²]

# ------------------------------------------------------------------
# System- och driftparametrar
# (Dessa är designpunktsvärden — justera fritt mellan körningar)
# ------------------------------------------------------------------

#: Temperaturen in i värmeväxlaren kommande ifrån 1 K pot [K]
T_varm_in: float = 1

#: Drifttemperatur för blandningskammaren  [K]
T_mc: float = 0.1

#: Molflöde av He-3 genom blandningskammaren  [mol/s]
n3_dot: float = 20e-6

#: He-3 molfraktion i inkommande ström (ren gas = 1.0)
x_gas: float = 1.0

#: Drifttemperatur för stillen  [K]
T_still: float = 0.8

#: Tryck vid pumpens utlopp  [Pa]
p_utlopp: float = 30_000 

#: Temperaturen vid pumpen, drift temperatur [K]
T_rum = 300

#: Radien för den inre röret av VVX, Antar att det är Tube in Tube VVX
r_vvx = 0.015

#: ytarea per längdenhet för vvx
p =  2*np.pi * r_vvx

