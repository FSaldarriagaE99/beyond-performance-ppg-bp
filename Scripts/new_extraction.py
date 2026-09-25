from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py


###### OBSOLETO
data_path = 'VitalDB_Train_Subset.h5'
data = {}
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T        

print(data["PPG"].shape)
data_ext = {}
data_ext["P1"] = data["PPG"][:100000]

ftext = Feature_Extraction(data_ext=data_ext)

demo_info = {}
demo_info["P1"] = {}
demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids = {}
segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids

features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
df = features_means["P1"].T
df.drop(columns=["segment_ID"], inplace=True)
df = df.T

df_fidu = fiducial_points["P1"].T
df_fidu.drop(columns=["segment_ID"], inplace=True)
df_fidu = df_fidu.T

data_ext["P1"] = data["PPG"][100000:200000]
ftext = Feature_Extraction(data_ext=data_ext)

demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
df2 = features_means["P1"].T
df2.drop(columns=["segment_ID"], inplace=True)
df2 = df2.T

df2_fidu = fiducial_points["P1"].T
df2_fidu.drop(columns=["segment_ID"], inplace=True)
df2_fidu = df2_fidu.T

data_ext["P1"] = data["PPG"][200000:300000]
ftext = Feature_Extraction(data_ext=data_ext)

demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
df3 = features_means["P1"].T
df3.drop(columns=["segment_ID"], inplace=True)
df3 = df3.T

df3_fidu = fiducial_points["P1"].T
df3_fidu.drop(columns=["segment_ID"], inplace=True)
df3_fidu = df3_fidu.T

data_ext["P1"] = data["PPG"][300000:400000]
ftext = Feature_Extraction(data_ext=data_ext)

demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
df4 = features_means["P1"].T
df4.drop(columns=["segment_ID"], inplace=True)
df4 = df4.T

df4_fidu = fiducial_points["P1"].T
df4_fidu.drop(columns=["segment_ID"], inplace=True)
df4_fidu = df4_fidu.T

data_ext["P1"] = data["PPG"][400000:]
ftext = Feature_Extraction(data_ext=data_ext)

demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids
features_means, features_medians, failed, fiducial_points = ftext.feature_extraction(save=False)
df5 = features_means["P1"].T
df5.drop(columns=["segment_ID"], inplace=True)
df5 = df5.T

df5_fidu = fiducial_points["P1"].T
df5_fidu.drop(columns=["segment_ID"], inplace=True)
df5_fidu = df5_fidu.T

with h5py.File('Features_VitalDB_Train_Subset.h5', 'w') as f:
    for g in data.keys():
        if g == "PPG" or g == "ABP":
            continue
        f.create_dataset(g, data=data[g].T)
    group = f.create_group("PPG_features")
    group.create_dataset("First_100k", data= df.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Second_100k", data= df2.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Third_100k", data= df3.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Fourth_100k", data= df4.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Last_data", data= df5.to_numpy(dtype=np.float64, na_value=np.nan))

with h5py.File('Fiducial_Points_VitalDB_Train_Subset.h5', 'w') as f:
    group = f.create_group("PPG_fiducial_points")
    group.create_dataset("First_100k", data= df_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Second_100k", data= df2_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Third_100k", data= df3_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Fourth_100k", data= df4_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
    group.create_dataset("Last_data", data= df5_fidu.to_numpy(dtype=np.float64, na_value=np.nan))