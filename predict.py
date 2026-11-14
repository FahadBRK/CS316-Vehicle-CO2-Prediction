"""Example: python predict.py --engine 2.0 --cylinders 4 --combined 8.0 --fuel X"""
import argparse
import json
import warnings
from pathlib import Path
import joblib
import pandas as pd
from src.features import NUMERIC, FUEL

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description="Estimate Canadian catalog rated CO2, in g/km.")
p.add_argument("--engine",type=float,required=True)
p.add_argument("--cylinders",type=int,required=True)
p.add_argument("--combined",type=float,required=True)
p.add_argument("--fuel",required=True,choices=["X","Z","D","E"])
args=p.parse_args()
meta=json.loads((root/"models/model_metadata.json").read_text())
if min(args.engine,args.cylinders,args.combined)<=0:
    p.error("Engine size, cylinder count and fuel consumption must be positive.")
if args.fuel not in meta["fuels_seen"]:
    p.error("This fuel was absent from the model's training data.")
values=[args.engine,args.cylinders,args.combined]
for feature,value in zip(NUMERIC,values):
    lo,hi=meta["training_ranges"][feature]
    if not lo<=value<=hi: warnings.warn(f"{feature} is outside the observed training range [{lo}, {hi}].")
row=pd.DataFrame([{**dict(zip(NUMERIC,values)),FUEL:args.fuel}])
# Only load trusted project artifacts: joblib uses Python pickle internally.
model=joblib.load(root/"models/selected_model.joblib")
print(f"Estimated rated CO2: {model.predict(row)[0]:.2f} g/km")
print("Scope: 2025 Canadian ratings. Real-world and lifecycle emissions are outside this model.")
