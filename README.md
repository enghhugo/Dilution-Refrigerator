# Dilution Refrigerator Simulation

Detta projekt är genomfört som en del av ett kandidatarbete vid Chalmers tekniska högskola, Institutionen för Mekanik och Maritima Vetenskaper.

Projektet syftar till att modellera och simulera ett utspädningskylare (dilution refrigerator), ett avancerat kylsystem som används för att uppnå extremt låga temperaturer inom bland annat kvantteknologi och lågtemperaturfysik.

---

## Syfte

Syftet med projektet är att:
- Modellera och simulera en utspädningskylare
- Analysera systemets termiska beteende
- Undersöka skalningsproblem och begränsningar i systemets prestanda

---

## Metod

Modellen är implementerad i Python med en objektorienterad struktur för att separera fysikaliska komponenter, konstanter och datahantering. Simuleringen körs i en Jupyter Notebook där resultaten analyseras och visualiseras.

---

## Hur man kör simuleringen

1. Ladda ner eller klona projektet
2. Bevara följande filstruktur:


```
DR/
│
├── DR_sim.ipynb
│
├── DR_OOP/
│   ├── __init__.py
│   ├── komponenter.py
│   ├── konstanter.py
│   └── termiskdata.py
│
└── data/
    ├── H_losning.csv
    ├── H_rent_He3.csv
    ├── Xv.csv
    ├── pv.csv
    ├── kemisk_potential4.csv
    └── mu40_He4.csv
```

3. Öppna `DR_sim.ipynb` i Jupyter Notebook eller VS Code
4. Kör alla celler i notebooken

---

## Projektstruktur

- **DR_sim.ipynb**  
  Huvudnotebook där simulering, analys och visualisering utförs.

- **DR_OOP/**
  - `komponenter.py` – Implementation av systemets fysiska komponenter och modell
  - `konstanter.py` – Fysiska konstanter som används i simuleringen
  - `termiskdata.py` – Hantering av termodynamiska och experimentella data

- **data/**  
  Innehåller tabeller och experimentella data som används i modellen

---

## Data

Modellen bygger på experimentella och tabellerade data för termodynamiska egenskaper hos heliumblandningar och relaterade systemparametrar.

---

## Resultat

Simuleringen används för att analysera kylsystemets temperaturutveckling och identifiera begränsningar i systemets prestanda vid olika driftförhållanden.

---

## Rapport

En mer detaljerad beskrivning av teori, metod och resultat finns i den tillhörande rapporten:

**Länk: [lägg in här]**

---

## Kommentar

Projektet är strukturerat enligt objektorienterade principer för att möjliggöra modulär utveckling och tydlig separation mellan fysik, data och simulering.
