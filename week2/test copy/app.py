import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os

# 🔹 **Dateipfade (für Hugging Face angepasst)**
MODEL_PATH = "price_prediction_model.pkl"
DATA_PATH = "precomputed_data.csv"

# 🔹 **Modell & Daten einmalig laden**
print("📥 Lade Modell...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Fehler: {MODEL_PATH} nicht gefunden!")

model = joblib.load(MODEL_PATH)

print("📥 Lade vorverarbeitete Daten...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"🚨 Fehler: {DATA_PATH} nicht gefunden!")

precomputed = pd.read_csv(DATA_PATH)

# 🔹 **Sicherstellen, dass `town` eindeutige Werte hat**
precomputed = precomputed.groupby('town', as_index=False).mean()

# 🔹 **Daten als Dictionary für schnellen Zugriff**
precomputed_dict = precomputed.set_index('town').to_dict(orient='index')

# 🔹 **Dropdown-Liste mit Gemeindenamen**
gemeinde_options = sorted(precomputed['town'].unique())

# 🔹 **Beschreibung der App**
beschreibung = """
🏡 **Wohnungspreis-Vorhersage mit Minergie & Steuerdaten**

🔹 Wähle **Wohnfläche (m²)**, **Zimmeranzahl** und **Gemeinde**  
🔹 Die App berechnet zwei Preise:
  - ✅ **Preis mit Minergie**
  - ⚡ **Preis ohne Minergie**

---

### 🔹 **Was ist Minergie?**
Minergie ist ein **Schweizer Baustandard** für **energieeffiziente Gebäude**.  
Gebäude mit Minergie-Standard verbrauchen **weniger Energie**,  
bieten **hohe Wohnqualität** und nutzen **erneuerbare Energien**.  

**Minergie-Label Varianten:**
- **Minergie** 🏡 → Standard für energieeffiziente Neubauten & Sanierungen  
- **Minergie-P** 🌱 → Besonders energieeffizient (vergleichbar mit Passivhäusern)  
- **Minergie-A** ☀️ → Gebäude, die mehr Energie produzieren als sie verbrauchen  
- **Minergie-ECO** 🌍 → Fokus auf Umweltverträglichkeit & gesundes Wohnen  

---

📌 **Datenquelle:**
Die Minergie-Daten wurden durch **Web-Scraping** von  
[minergie.ch](https://www.minergie.ch/de/gebaeude/gebaeudeliste/?canton=zh&country=&zip_place=&street_nr=&gid=&typeofuse=&constructiontype=&year=&sortby=year_desc&numres=12&p=50)  
extrahiert, bereinigt und in das Modell integriert.
"""

# 🔹 **Vorhersagefunktion**
def predict_prices(area, rooms, town):
    if town not in precomputed_dict:
        return "🚨 Fehler: Keine Daten für diese Gemeinde verfügbar."

    # Werte aus dem Dictionary abrufen
    data = precomputed_dict[town]

    # 🔹 **Fehlende Features mit Standardwerten setzen**
    missing_features = ["pop_dens", "emp"]
    for feature in missing_features:
        data.setdefault(feature, 0)

    # 🔹 **Modell-Eingaben vorbereiten**
    input_with_minergie = np.array([[area, rooms, data['lat'], data['lon'], 
                                     data['minergie_anteil'], data['tax_income'],
                                     data['pop_dens'], data['emp'],  
                                     data['price_per_sqm'], data['avg_rent']]])

    input_without_minergie = input_with_minergie.copy()
    input_without_minergie[0][4] = 0  # Minergie-Anteil auf 0 setzen

    # 🔹 **Modell-Vorhersage**
    predicted_price_with = model.predict(input_with_minergie)[0]
    predicted_price_without = model.predict(input_without_minergie)[0]

    return (f"✅ Preis mit Minergie: **CHF {predicted_price_with:,.2f}**\n"
            f"⚡ Preis ohne Minergie: **CHF {predicted_price_without:,.2f}**")

# 🔹 **Gradio Interface**
iface = gr.Interface(
    fn=predict_prices,
    inputs=[
        gr.Number(label="Wohnfläche (m²)", value=50),
        gr.Number(label="Zimmeranzahl", value=2),
        gr.Dropdown(choices=gemeinde_options, label="Gemeinde")
    ],
    outputs="text",
    title="🏡 Wohnungspreis-Vorhersage mit Minergie & Steuerdaten",
    description=beschreibung,
    theme="huggingface",
    live=False  # Verhindert unnötige Neuberechnungen bei Eingaben
)

# 🔹 **App starten**
iface.launch()
