# Dieser Code liest die JSON-Datei und wandelt die Regeln in Zahlen (Features) um, die Ihr Machine Learning Modell versteht
from pathlib import Path
import json
import pandas as pd
import numpy as np


class RacingIntelligenceEngine:
    def __init__(self, knowledge_base_path: str | None = None):
        if knowledge_base_path is None:
            knowledge_base_path = Path(__file__).with_suffix("").parent / "racing_criteria.json"
        else:
            knowledge_base_path = Path(knowledge_base_path)

        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)


    def analyze_driver_eligibility(self, driver_data):
        """
        Prüft Reglements für F1, F2, F3.
        """
        features = {}
        rules = self.kb['series_rules']
        
        # --- F1 Checks ---
        f1 = rules['f1']
        # Hard Constraints
        sl_ok = driver_data.get('superlicense_points', 0) >= f1['requirements']['min_superlicense_points']
        age_ok = driver_data.get('age', 0) >= f1['requirements']['min_age']
        exp_ok = driver_data.get('junior_series_years', 0) >= f1['requirements']['junior_series_experience_years']
        
        features['f1_qualified'] = 1 if (sl_ok and age_ok and exp_ok) else 0
        
        # Marketing Boost (USA/China)
        nat = driver_data.get('nationality', '')
        features['f1_marketing_boost'] = 1 if nat in f1['market_value_boost']['high_value_nations'] else 0
        
        # Risk Factor (Unruhe)
        behavior = driver_data.get('social_media_behavior', 'neutral')
        features['f1_risk_factor'] = 1 if behavior in f1['market_value_boost']['negative_traits'] else 0

        # --- F2 Checks ---
        f2 = rules['f2']
        # F3 Stagnation (Mehr als 2 Jahre in F3 ist schlecht)
        f3_years = driver_data.get('years_in_f3', 0)
        features['f2_stagnation_penalty'] = 1 if f3_years > f2['requirements']['max_years_in_f3'] else 0
        
        # Anti-Champion Rule
        is_champ = driver_data.get('is_f2_champion', False)
        features['f2_banned'] = 1 if is_champ else 0

        # --- F3 Checks ---
        f3 = rules['f3']
        prev_series = driver_data.get('previous_series', '')
        # Pathway Score: 1.0 (Ideal), 0.5 (Okay), 0.0 (Risky/F4 direct)
        if prev_series in f3['requirements']['preferred_pathway']:
            features['f3_pathway_score'] = 1.0
        elif prev_series in f3['requirements']['risky_pathway']:
            features['f3_pathway_score'] = 0.0
        else:
            features['f3_pathway_score'] = 0.5

        return features

    def analyze_driver_biometrics(self, driver_data):
        """
        Vergleicht Fahrer mit den idealen physikalischen Attributen.
        Gibt 'Fit-Scores' zurück (1.0 = perfekt, niedriger = Abweichung).
        """
        features = {}
        bio = self.kb['driver_profile']['biometrics']
        career = self.kb['driver_profile']['career']

        # Gewichts-Check (Ist Fahrer im Fenster?)
        w = driver_data.get('weight_kg', 70)
        if bio['ideal_weight_min_kg'] <= w <= bio['ideal_weight_max_kg']:
            features['phys_weight_score'] = 1.0
        else:
            # Einfache Distanzberechnung (Penalty für jedes kg daneben)
            dist = min(abs(w - bio['ideal_weight_min_kg']), abs(w - bio['ideal_weight_max_kg']))
            features['phys_weight_score'] = max(0, 1.0 - (dist * 0.1))

        # Halsumfang (G-Force Resistenzen)
        neck = driver_data.get('neck_cm', 42)
        features['phys_neck_strength'] = 1.0 if neck >= bio['neck_circumference_min_cm'] else 0.5

        # Alter (Peak Performance Window)
        age = driver_data.get('age', 20)
        if career['peak_performance_age_start'] <= age <= career['peak_performance_age_end']:
            features['age_peak_window'] = 1
        else:
            features['age_peak_window'] = 0
            
        return features

    def analyze_team_fit(self, team_data, driver_data):
        """
        Kombiniert Team-Eigenschaften mit Fahrer-Ressourcen.
        """
        features = {}
        t_prof = self.kb['team_profile']
        
        # Politische Macht (Veto Recht Teams haben Vorteile bei Regeln)
        team_name = team_data.get('team_name', '')
        features['team_political_power'] = 1 if team_name in t_prof['political_power']['has_veto_right'] else 0
        
        # Budget Match: Hat der Fahrer genug Startkapital für die Liga?
        # (Hier vereinfachtes Beispiel für F1 Kosten)
        driver_capital = driver_data.get('sponsor_capital_chf', 0)
        features['financial_viability'] = min(driver_capital / 8000000, 1.0) # Normiert auf das ideale Startkapital

        return features

    def analyze_vehicle_telemetry(self, vehicle_data):
        """
        Wandelt rohe Fahrzeugdaten in Status-Flags um.
        """
        features = {}
        v_data = self.kb['vehicle_telemetry']
        
        # Damage Flags (One-Hot Encoding für das Modell)
        features['vehicle_engine_damaged'] = 1 if vehicle_data.get('engine_status') == 'damaged' else 0
        features['vehicle_tire_damaged'] = 1 if vehicle_data.get('tire_status') == 'damaged' else 0
        
        # Aerodynamik Check
        drs = vehicle_data.get('drs_active', False)
        features['aero_mode_attack'] = 1 if drs else 0
        
        return features

    def generate_full_profile(self, driver, team, vehicle):
        """
        Wrapper-Funktion, die ALLES zusammenführt.
        Dies ist der Input Vektor für Ihr ML Modell.
        """
        full_vector = {}
        full_vector.update(self.analyze_driver_eligibility(driver))
        full_vector.update(self.analyze_driver_biometrics(driver))
        full_vector.update(self.analyze_team_fit(team, driver))
        full_vector.update(self.analyze_vehicle_telemetry(vehicle))
        return full_vector

# --- BEISPIEL ANWENDUNG ---

if __name__ == "__main__":
    # Engine starten
    engine = RacingIntelligenceEngine()

    # Dummy Daten (Wie sie aus Ihrer Datenbank kommen würden)
    driver_input = {
        "name": "Rookie One",
        "age": 19,
        "nationality": "USA",           # Bonus!
        "weight_kg": 72,                # Ideal
        "neck_cm": 38,                  # Zu schwach (<40)
        "superlicense_points": 30,      # Nicht genug für F1
        "years_in_f3": 1,
        "previous_series": "F4",        # Riskanter Sprung für F3
        "sponsor_capital_chf": 9000000  # Gutes Startkapital
    }

    team_input = {
        "team_name": "Ferrari",         # Politischer Bonus
        "budget": "High"
    }

    vehicle_input = {
        "engine_status": "ok",
        "drs_active": True,
        "tire_status": "damaged"        # Problem
    }

    # Features generieren
    ml_features = engine.generate_full_profile(driver_input, team_input, vehicle_input)

    print("=== Generierte Features für das ML Modell ===")
    print(json.dumps(ml_features, indent=2))