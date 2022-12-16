import numpy as np
import tsfeatures as tsf
import pandas as pd
import math
import os
import glob
from tqdm import tqdm


def compress(ts, variable_name):
    interval = tsf.intervals(ts)
    stl = tsf.stl_features(ts)
    # Mean
    mean = interval["intervals_mean"]
    # Variance
    sd = interval["intervals_sd"]
    variance = 0
    if not math.isnan(sd):
        variance = sd * sd
    # ACF1 - First order of auto-correlation.
    acf1 = tsf.acf_features(ts)['x_acf1']
    # Strength of trend
    trend = stl['trend']
    # Strength of linearity
    linearity = stl['linearity']
    # Strength of curvature
    curvature = stl['curvature']
    # Strength of seasonality
    season = stl['seasonal_period']
    # Strength of entropy
    entropy = tsf.entropy(ts)['entropy']
    # Lumpiness - Changing variance in remainder.
    lumpiness = tsf.lumpiness(ts)['lumpiness']
    # Strength of spikiness
    spikiness = stl['spike']
    # Fspots - Flat spots using disretization
    fspots = tsf.flat_spots(ts)['flat_spots']
    # Cpoints - The number of crossing points
    cpoints = tsf.crossing_points(ts)['crossing_points']
    return dict({
        variable_name + "_mean": mean,
        variable_name + "_variance": variance,
        variable_name + "_acf1": acf1,
        variable_name + "_trend": trend,
        variable_name + "_linearity": linearity,
        variable_name + "_curvature": curvature,
        variable_name + "_season": season,
        variable_name + "_entropy": entropy,
        variable_name + "_lumpiness": lumpiness,
        variable_name + "_spikiness": spikiness,
        variable_name + "_fspots": fspots,
        variable_name + "_cpoints": cpoints
    })


def dimensionality_deduction(data, address):
    result = {}
    for feature in data:
        result.update(compress(data[feature].values, feature))
    result['address'] = address
    return result


def run_deduction(files, feature_path):
    features = []
    for f in tqdm(files):
        print("Start to process file:", f)
        address = os.path.splitext(os.path.basename(f))[0]
        data = pd.read_csv(f, dtype=np.float)
        if len(data) <= 1:
            print("Too few data to process -> skip file", f)
            continue
        features.append(dimensionality_deduction(data, address))
    pd.DataFrame.from_records([s for s in features]).to_csv(feature_path, index=False)


def run():
    ponzi_contract_ts_files = glob.glob(os.path.join("data", "timeseries", "ponzi", "*.csv"))
    ponzi_feature_path = os.path.join("features", "timedependent", "PonziTimeDependentFeatures.csv")
    run_deduction(files=ponzi_contract_ts_files, feature_path=ponzi_feature_path)

    dapp_contract_ts_files = glob.glob(os.path.join("data", "timeseries", "nonPonzi", "*.csv"))
    dapp_feature_path = os.path.join("features", "timedependent", "NonPonziTimeDependentFeatures.csv")
    run_deduction(files=dapp_contract_ts_files, feature_path=dapp_feature_path)


if __name__ == '__main__':
    run()
