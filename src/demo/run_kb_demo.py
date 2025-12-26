from knowledge_base.racing_intelligence_engine import RacingIntelligenceEngine

def run_demo():
    engine = RacingIntelligenceEngine()

    driver_input = {
        "age": 20,
        "nationality": "USA",
        "superlicense_points": 25,
        "junior_series_years": 2,
        "years_in_f3": 1,
        "previous_series": "FRECA",
        "social_media_behavior": "neutral",
        "weight_kg": 70,
        "neck_cm": 41,
        "sponsor_capital_chf": 9_000_000,
    }

    team_input = {"team_name": "Ferrari"}
    vehicle_input = {"engine_status": "ok", "drs_active": True, "tire_status": "ok"}

    kb_features = engine.generate_full_profile(
        driver_input, team_input, vehicle_input
    )

    print("Knowledge Base Features")
    for k, v in kb_features.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_demo()
