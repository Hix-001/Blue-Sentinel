import pandas as pd
import random

def generate_historical_data():
    vessels = []
    flags = ["India", "Sri Lanka", "China", "Panama", "Liberia", "Unknown"]
    types = ["Cargo", "Tanker", "Fishing", "Passenger", "Trawler"]
    
    for i in range(50):
        vessels.append({
            "MMSI": random.randint(200000000, 700000000),
            "Name": f"REAL_VESSEL_{i+100}",
            "Lat": random.uniform(5.0, 15.0),
            "Lon": random.uniform(70.0, 85.0),
            "SOG": random.uniform(0.0, 20.0),
            "COG": random.uniform(0, 360),
            "Type": random.choice(types),
            "Flag": random.choice(flags)
        })
    
    df = pd.DataFrame(vessels)
    df.to_csv("historical_ais.csv", index=False)

if __name__ == "__main__":
    generate_historical_data()