"""
termiskdata.py
==============

Läser in termodynamiska datatabeller från CSV-filer och exponerar dem
som anropsbara interpolatorer via klassen ``TermiskData``.

Alla interpolatorer byggs upp med lazy loading — CSV-filer läses endast
in när motsvarande egenskap används för första gången. Detta innebär att
``FileNotFoundError`` aldrig uppstår vid import av modulen; felet skjuts
i stället upp tills data faktiskt efterfrågas.

Förväntat CSV-format
--------------------
Varje fil ska innehålla en rektangulär eller endimensionell tabell där:

- första kolumnen innehåller temperatur T [K]
- första raden innehåller molfraktion X för He-3 [-]
  (eller endast temperatur för 1D-tabeller)
- övriga celler innehåller den tabellerade storheten

Tabellförteckning (Radebaugh, 1967)
-----------------------------------
FIL_H_LOS_He3_4 :
    Entalpi för He-3 i utspädd lösning, H(T, X)
    (Tabell 9) [J/mol]

FIL_H_REN_He3 :
    Entalpi för rent He-3, H°(T)
    (Tabell 4) [J/mol]

FIL_XV :
    He-3-molfraktion i ångfas, X_v(T, X)
    (Tabell 15) [-]

FIL_PV :
    Totalt ångtryck, p_v(T, X)
    (Tabell 14) [Torr]

FIL_MU4 :
    Kemisk potential för rent helium-4
    (Tabell 5) [J/mol]

FIL_U4 :
    Termen representerar avvikelsen i kemisk potential mellan
    rent He-4 och He-4 i He-3/He-4-lösningen.
    (Tabell 10) [J/mol]
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d

from .konstanter import DATA_MAPP


# ------------------------------------------------------------------
# Standardsökvägar till datafiler
# (kan åsidosättas genom att skicka in egna sökvägar till TermiskData)
# ------------------------------------------------------------------

_STANDARDFILER: dict[str, Path] = {
    "h_los":  DATA_MAPP / "H_losning.csv",       
    "h_ren":  DATA_MAPP / "H_rent_He3.csv",
    "xv":     DATA_MAPP / "Xv.csv",
    "pv":     DATA_MAPP / "pv.csv",
    "mu40": DATA_MAPP / "mu40_He4.csv",
    "U4": DATA_MAPP / "kemisk_potential4.csv"
}


# ------------------------------------------------------------------
# Hjälpfunktioner för filläsning och interpolation
# ------------------------------------------------------------------

def _las_fil(sökväg: Path) -> pd.DataFrame:
    """
    Läser in en csv och returnerar en pandas DataFrame.
    Ger ett tydligt felmeddelande om filen saknas.
    
    Radindex blir temperaturer, kolumnrubriker blir molfraktioner.
    """
    if not sökväg.exists():
        raise FileNotFoundError(
            f"Datafil saknas: {sökväg}\n"
            "Kontrollera att filen finns i data/-mappen och att "
            "filnamnet stämmer överens med inställningarna i _STANDARDFILER."
        )
    return pd.read_csv(sökväg, index_col=0, header=0)


def _bygg_1d_interpolator(sokväg: Path) -> Callable[[float], float]:
    """
    Bygger en linjär 1D-interpolator  f(T)  från en enkelkolumns Excel-fil.
    
    Används för tabeller som bara beror på temperaturen, t.ex. H°(T) för rent He-3.
    Linjär interpolation innebär att värden mellan tabellpunkterna uppskattas
    med en rät linje mellan närmaste grannar.
    """
    df = _las_fil(sokväg)
    T = df.index.values.astype(float)       # temperaturer som NumPy-array
    y = df.iloc[:, 0].values.astype(float)  # värden från första kolumnen
    
    # interp1d från SciPy skapar interpolatorfunktionen.
    # bounds_error=True ger tydligt fel om man frågar utanför tabellens intervall.
    _f = interp1d(T, y, kind="linear", bounds_error=True)

    def funk(T_värde: float) -> float:
        return float(_f(T_värde))

    return funk


def _bygg_2d_interpolator(sokväg: Path) -> Callable[[float, float], float]:
    """
    Bygger en kubisk 2D-interpolator  f(T, X)  från en 2D Excel-tabell.
    
    Används för tabeller som beror på både temperatur och molfraktion,
    t.ex. entalpin H(T, X) eller ångtrycket p(T, X).
    
    Kubisk spline ger en mjukare kurva än linjär interpolation, vilket
    passar bättre för termodynamiska data som varierar jämnt.
    """
    df = _las_fil(sokväg)
    T = df.index.values.astype(float)    # temperaturer längs raderna
    X = df.columns.values.astype(float) # molfraktioner längs kolumnerna
    Z = df.values.astype(float)         # tabellvärden som 2D-matris

    # RectBivariateSpline från SciPy skapar interpolatorn för rektangulära gitter.
    # kx=3, ky=3 anger kubisk (tredje gradens) spline i båda riktningarna.
    _f = RectBivariateSpline(T, X, Z, kx=3, ky=3)

    def funk(T_värde: float, X_värde: float) -> float:
        # grid=False säger att vi vill ha ett enskilt värde, inte en hel matris
        return float(_f(T_värde, X_värde, grid=False))

    return funk


# ------------------------------------------------------------------
# TermiskData — huvudklassen
# ------------------------------------------------------------------

class TermiskData:
    """
    Behållare för alla termodynamiska uppslagstabeller i DR-simuleringen.

    Interpolatorerna byggs upp latent via ``cached_property``, vilket innebär
    att Excel-filerna bara läses in vid första användningen av respektive egenskap.

    Parametrar
    ----------
    filer : dict[str, Path], valfritt
        Åsidosätt någon av standardsökvägarna. Nycklarna är desamma som
        i ``_STANDARDFILER``: ``"h_los"``, ``"h_ren"``, ``"xv"``, ``"pv"``.

    Exempel
    -------
    >>> tabeller = TermiskData()
    >>> tabeller.H_ren(0.5)           # entalpi för rent He-3 vid 0.5 K  [J/mol]
    >>> tabeller.H_losning(0.1, 0.065) # entalpi i lösning vid 0.1 K, X=0.065
    >>> tabeller.Xv(0.8, 0.10)        # ångsammansättning vid stillförhållanden
    >>> tabeller.pv(0.8, 0.10)        # ångtryck vid stillförhållanden  [Pa]
    """

    def __init__(self, filer: dict[str, Path] | None = None) -> None:
        # Slå ihop standardfilerna med eventuella anpassade sökvägar
        self._filer: dict[str, Path] = {**_STANDARDFILER, **(filer or {})}

    # ---- 1D-interpolatorer -----------------------------------------------

    @cached_property
    def H_ren_HE3(self) -> Callable[[float], float]:
        """
        Molar entalpi för rent He-3, H°(T)  [J/mol].
        
        Byggs från en enkelkolumnstabell med temperatur som index.
        cached_property innebär att filen bara läses in en gång —
        därefter returneras den sparade interpolatorn direkt.
        """
        return _bygg_1d_interpolator(self._filer["h_ren"])

    # ---- 2D-interpolatorer -----------------------------------------------

    @cached_property
    def H_losning_HE3_4(self) -> Callable[[float, float], float]:
        """
        Molar entalpi för He-3 i utspädd lösning, H(T, X)  [J/mol].

        Parametrar
        ----------
        T : float  Temperatur [K]
        X : float  He-3 molfraktion i utspädd fas [-]
        """
        return _bygg_2d_interpolator(self._filer["h_los"])

    @cached_property
    def Xv(self) -> Callable[[float, float], float]:
        """
        He-3 molfraktion i ångfasen, X_v(T, X)  [-].
        (Radebaugh 1967, Tabell 15)

        Parametrar
        ----------
        T : float  Temperatur [K]
        X : float  He-3 molfraktion i vätskefasen [-]
        """
        return _bygg_2d_interpolator(self._filer["xv"])

    @cached_property
    def pv(self) -> Callable[[float, float], float]:
        """
        Totalt ångtryck, p_v(T, X)  [Pa].
        (Radebaugh 1967, Tabell 14)

        Parametrar
        ----------
        T : float  Temperatur [K]
        X : float  He-3 molfraktion i vätskefasen [-]
        """
        return _bygg_2d_interpolator(self._filer["pv"])
    
    @cached_property
    def osmotiskt_tryck(self) -> Callable[[float, float], float]:
        """
        
        """
        return _bygg_2d_interpolator(self._filer["osmot"])
    
    @cached_property
    def mu40_He4(self) -> Callable[[float], float]:
        """
        Kemisk potential för rent He-4 vid T=0 som referenspunkt, μ°_4(T)  [J/mol].
        (Radebaugh 1967, Tabell 5)
        """
        return _bygg_1d_interpolator(self._filer["mu40"])
    
    @cached_property
    def kemisk_potential4(self) -> Callable[[float, float], float]:
        """
        Kemisk potential för He-4  [J/mol].
        (Radebaugh 1967, Tabell 10)
        """
        return _bygg_2d_interpolator(self._filer["U4"])

# ------------------------------------------------------------------
# Standardinstans på modulnivå
# ------------------------------------------------------------------

#: Delad standardinstans — importera och använd direkt om du inte behöver
#: anpassade filsökvägar:
#:
#:     from DR_OOP.termiskdata import standardtabeller
#:     h = standardtabeller.H_ren(0.5)
standardtabeller = TermiskData()
