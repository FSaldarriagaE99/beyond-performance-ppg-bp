from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py
from bp_lgbm.local_paths import PULSE_DB_SUP_DIR

data_path = PULSE_DB_SUP_DIR / "VitalDB_Train_Subset.h5"
#data_path = 'VitalDB_Train_Subset.h5'
data = {}
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T        

print(data["PPG"].shape)
data_ext = {}

splits = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, len(data["PPG"])]

features_list = list()
fiducials_list = list()

for i in range(0, len(splits)-1):
    
    data_ext["P1"] = data["PPG"][splits[i]:splits[i+1]]

    ftext = Feature_Extraction(data_ext=data_ext)

    demo_info = {}
    demo_info["P1"] = {}
    demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
    print(demo_info["P1"]["SamplingFrequency"])
    ftext.demo_info = demo_info

    segment_ids = {}
    segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
    ftext.segment_ids = segment_ids

    features_means, features_medians, fiducial_points = ftext.feature_extraction(save=False)
    df = features_means["P1"].T
    
    df.index = df["segment_ID"]
    df.drop(columns=["segment_ID"], inplace=True)
    df = df.T

    # Convert into dictionaries and then append it
    dict_feat = df.to_dict(orient="list")
    features_list.append(dict_feat)

    df_fidu = fiducial_points["P1"].T
    df_fidu.drop(columns=["segment_ID"], inplace=True)
    df_fidu = df_fidu.T

    # Convert into dictionaries and then append it
    dict_fid = df_fidu.to_dict(orient="list")
    fiducials_list.append(dict_fid)

features = df.index.tolist()
fiducials = df_fidu.index.tolist()

# later: rebuild DataFrames and concat
dfs_feat = [pd.DataFrame(d) for d in features_list]
final_feat_df = pd.concat(dfs_feat, ignore_index=True, axis=1)
print(final_feat_df.head())
print(final_feat_df.shape)

# later: rebuild DataFrames and concat
dfs_fid = [pd.DataFrame(d) for d in fiducials_list]
final_fid_df = pd.concat(dfs_fid, ignore_index=True, axis=1)
print(final_fid_df.head())
print(final_fid_df.shape)

final_fid_df = final_fid_df.replace({pd.NA: np.nan})

with h5py.File('Features_VitalDB_Train_Subset.h5', 'w') as f:
    for g in data.keys():
        if g == "PPG" or g == "ABP":
            continue
        f.create_dataset(g, data=data[g].T)
    dst = f.create_dataset("PPG_features", data= final_feat_df.to_numpy(dtype=np.float64, na_value=np.nan))
    dst.attrs["Feature_Names"] = np.array(features, dtype="S")

with h5py.File('Fiducial_Points_VitalDB_Train_Subset.h5', 'w') as f:
    dst = f.create_dataset("PPG_fiducial_points", data= final_fid_df.to_numpy(dtype=np.float64, na_value=np.nan))
    dst.attrs["Fiducial_points"] = np.array(fiducials, dtype="S")