from fearture_extraction import Feature_Extraction
import numpy as np
import pandas as pd
import h5py

####### Only Fiducial Extraction for the Test Subset ########
data_path = 'VitalDB_CalFree_Test_Subset.h5'
data = {}
with h5py.File(data_path, 'r') as f:
    for group_name in f:
        group = f[group_name]
        data[group_name] = group[()].T

print(data["PPG"].shape)
data_ext = {}
data_ext["P1"] = data["PPG"][:]

ftext = Feature_Extraction(data_ext=data_ext)

demo_info = {}
demo_info["P1"] = {}
demo_info["P1"]["SamplingFrequency"] = data["SF"][0][0] # 125 Hz
print(demo_info["P1"]["SamplingFrequency"])
ftext.demo_info = demo_info

segment_ids = {}
segment_ids["P1"] = np.arange(1,len(data_ext["P1"])+1)
ftext.segment_ids = segment_ids

fiducial_points = ftext.only_fiducials()
df_fidu = fiducial_points["P1"].T
df_fidu.drop(columns=["segment_ID"], inplace=True)
df_fidu = df_fidu.T
df_fidu = df_fidu.replace({pd.NA: np.nan})

with h5py.File('Fiducial_Points_VitalDB_CalFree_Test_Subset.h5', 'w') as f:
    for k in data.keys():
        f.create_dataset(k, data=data[k].T)
    fiducials = f.create_dataset("PPG_fiducial_points", data= df_fidu.to_numpy(dtype=np.float64, na_value=np.nan))
    fiducials.attrs["Fiducial_points"] = np.array(df_fidu.index, dtype="S")