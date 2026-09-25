import h5py
import numpy as np
import pandas as pd
import scipy.stats
from dotmap import DotMap

from pyPPG import PPG, Fiducials, Biomarkers
##Modified (Fiducials2 and Biomarkers2 are the same as the originals)
from lib_changes import PPG2, Fiducials2, Biomarkers2

import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from lib_changes import fiducials2 as FP2 ##Modified

import pyPPG.biomarkers as BM
from lib_changes import biomarkers2 as BM2 ##Modified

from other_functions_PPG import Others ### Class with some other functions

class Feature_Extraction():
    def __init__(self, h5name: str = "", csvname: str = "", ids: dict = {}, demo_info: dict = {}, data_ext: dict = None, data_path: str = ""):
        
        self.demo_info = demo_info.copy()
        
        self.signal_dict = {}
        self.mean = {}
        self.median = {}

        self.filename_save = h5name
        self.filename_csv = csvname

        # Opens the archive read mode only with h5py
        data = {}
        if data_path != "":
            with h5py.File(data_path, 'r') as f:
                for name in f:
                    obj = f[name]
                    if isinstance(obj, h5py.Dataset):
                        data[name] = obj[()].T
                    elif isinstance(obj, h5py.Group):
                        group = f[name]
                        for dst in group:
                            data[f"{name}_{dst}"] = group[dst][()].T
            self.data = data.copy()
        elif data_ext is not None:
            self.data = data_ext.copy()
        
        if ids == {}:
            optional_ids = {}
            data = self.data.copy()
            for k in data.keys():
                optional_ids[k] = np.arange(1,len(data[k])+1)
            self.segment_ids = optional_ids.copy()
        else:
            self.segment_ids = ids.copy()

    def  save_h5(self, fiducials: dict, means: dict, medians: dict, fiducials_names: list, filename: str):
        demo_info = self.demo_info.copy()
        samples = self.samples.copy()
        with h5py.File(filename, 'w') as f:
            ### Creates a group per patient (or for the keys in fiducials)
            for patient_id, fiducials_list in fiducials.items():
                grp = f.create_group(patient_id)
                # Saves the demographic info in the attributes of each group
                if demo_info is not None:
                    for attr_name, attr_value in demo_info[patient_id].items():
                        grp.attrs[attr_name] = attr_value
                fiducials_list = fiducials_list.replace({pd.NA: np.nan})
                ### Creates 3 datasets: segments,mean_patient and median_patient
                ### the attributes of each dataset are their respective columns
                dset = grp.create_dataset(f'Fiducial_points', data=fiducials_list.to_numpy(dtype = float, na_value = np.nan))
                dset.attrs['fiducial_order'] = np.array(fiducials_names, dtype= 'S')
                # As an additional attribute for "segments" the number of samples were added for each signal
                dset.attrs["N-samples"] = samples[patient_id]
                stats1 = means[patient_id]
                me = grp.create_dataset("Features_means",data=stats1.to_numpy(dtype = float, na_value = np.nan))
                me.attrs["features"] = np.array(stats1.index.tolist(), dtype= 'S')
                stats2 = medians[patient_id]
                med = grp.create_dataset("Features_medians",data=stats2.to_numpy(dtype = float, na_value = np.nan))
                med.attrs["features"] = np.array(stats2.index.tolist(), dtype= 'S')

    def stats_features(self, features: pd.DataFrame):
        X = pd.DataFrame(columns=features.columns)
        Med = pd.DataFrame(columns=features.columns)
        for ft in features.columns:
            X.loc[0,ft] = features[ft].mean()
            Med.loc[0,ft] = features[ft].median()

        return X, Med
    
    def only_fiducials(self):
        data = self.data.copy()
        segment_ids = self.segment_ids.copy()
        demo_info = self.demo_info.copy()
        signal_dict = self.signal_dict.copy()

        for i in data.keys():
            signal = DotMap()
            signal.filtering = True # whether or not to filter the PPG signal
            signal.fL=0.5000001 # Lower cutoff frequency (Hz)
            signal.fH=12 # Upper cutoff frequency (Hz)
            signal.order=4 # Filter order
            signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"
        
            # Initialise the correction for fiducial points
            corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
            correction=pd.DataFrame()
            correction.loc[0, corr_on] = True
            signal.correction=correction

            #Initialise cycling storage variables
            fp_pt = pd.DataFrame()
            fp_pt_list = []

            print(f"Feature Extraction of: {i}")
            for sig in np.arange(len(data[i])): # Processing each signal
                
                signal.name = sig
                signal.start_sig = 0
                signal.end_sig = len(data[i][sig])
                signal.v = data[i][sig]
                signal.fs = int(demo_info[i]["SamplingFrequency"])

                #### Preprocess the signal with pyPPG (filtering and acquires the derivatives)
                prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
                signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

                # Create a PPG class
                s = PPG2(signal)

                # Acquire the fiducial points
                fpex = FP2.FpCollection(s=s)
                fiducials = fpex.get_fiducials(s=s)
                df = pd.DataFrame(fiducials)
                # Saving the extracted fiducials (flattened so each row will be a singular signal)
                df = df.values.flatten()
                fp_pt_list.append(df)

            fp_pt = pd.DataFrame(fp_pt_list)
            fp_pt.insert(0,"segment_ID",segment_ids[i])
            signal_dict[i] = fp_pt.T
        
        self.fiducial_ponts = signal_dict.copy()

        return signal_dict

    def feature_extraction(self, save: bool = True):
        data = self.data
        segment_ids = self.segment_ids
        demo_info = self.demo_info

        samples = {}
        signal_dict = self.signal_dict
        mean = self.mean
        median = self.median
        
        for i in data.keys():
            signal = DotMap()
            signal.filtering = True # whether or not to filter the PPG signal
            signal.fL=0.5000001 # Lower cutoff frequency (Hz)
            signal.fH=12 # Upper cutoff frequency (Hz)
            signal.order=4 # Filter order
            signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"
        
            # Initialise the correction for fiducial points
            corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
            correction=pd.DataFrame()
            correction.loc[0, corr_on] = True
            signal.correction=correction

            #Initialise cycling storage variables
            fp_pt = pd.DataFrame()
            ft_pt_mean = pd.DataFrame()
            ft_pt_median = pd.DataFrame()
            fp_pt_list = []
            fp_col = []
            samples[i] = []

            print(f"Feature Extraction of: {i}")
            
            for sig in np.arange(len(data[i])): # Processing each signal
                
                signal.name = sig
                signal.start_sig = 0
                signal.end_sig = len(data[i][sig])
                signal.v = data[i][sig]
                signal.fs = int(demo_info[i]["SamplingFrequency"])

                #### Preprocess the signal with pyPPG (filtering and acquires the derivatives)
                prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
                signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

                # Create a PPG class
                s = PPG2(signal)

                # Acquire the fiducial points
                fpex = FP2.FpCollection(s=s)
                fiducials = fpex.get_fiducials(s=s)

                # Create a fiducials class
                fp = Fiducials(fp=fiducials)

                # Just saving the names of the fiducials for later use in the h5 file
                df = pd.DataFrame(fiducials)
                if  sig == 0:
                    samples[i].append(len(s.ppg))
                    fp_col = df.columns
                else:
                    pass
            
                # Saving the extracted fiducials (flattened so each row will be a singular signal)
                df = df.values.flatten()
                fp_pt_list.append(df)

                # Init the biomarkers package (the features)
                # Using the modified version for personal selected features
                bmex = BM2.BmCollection2(s=s, fp=fp)

                # Extract biomarkers
                bm_defs, bm_vals = bmex.get_biomarkers2(get_stat=False)
                bm_df = bm_vals["ppg_features"]
                df = bm_df.drop(columns="TimeStamp")
                Tpp = bm_df["Tpp"]
                
                # PRV (pulse rate variability), we need the data of the whole segment to calculate the PRV (change in the time between beats)
                # The array needs to be bigger have more than 1 element or it wont work
                if len(Tpp) > 0:
                    PRV = np.diff(Tpp, prepend=Tpp[0])
                    PRV = PRV*1000 # to ms
                    df["PRV"] = PRV 
                    # interquartil range, standart deviation of PRV
                    sdPRV = pd.Series(PRV).std()
                    IQR = scipy.stats.iqr(PRV)
                else:
                    df["PRV"] = np.nan
                    sdPRV = np.nan
                    IQR = np.nan

                # Instead of saving the values of the features (since we have values of the features by beat to beat windows)
                # we will save the mean and median of the features as the finale feature datasets.
                x,y = self.stats_features(df)

                # Then we will add some additional features that are obtained not from windows but from the whole signal/segment of the patient
                # Adding Kurtosis and Skewness of the whole segment to the finale features dataset
                k = scipy.stats.kurtosis(s.ppg)
                sk = scipy.stats.skew(s.ppg)
                x["FullKurt"], y["FullKurt"] = [k,k]
                x["FullSkew"], y["FullSkew"] = [sk,sk]
                # Adding the PRV stadistics of the whole segement to the finale features dataset
                x["sdPRV"], y["sdPRV"] = [sdPRV,sdPRV]
                x["IQR_PRV"], y["IQR_PRV"] = [IQR,IQR]
                
                ft_pt_mean = pd.concat([ft_pt_mean,x])
                ft_pt_median = pd.concat([ft_pt_median,y])
                
            fp_pt = pd.DataFrame(fp_pt_list)
            fp_pt.insert(0,"segment_ID",segment_ids[i])
            signal_dict[i] = fp_pt.T
            fp_col.insert(0,"segment_ID")
            # Saving the mean and median of the features per signal for each patient
            ft_pt_mean.insert(0,"segment_ID",segment_ids[i])
            mean[i] = ft_pt_mean.T
            ft_pt_median.insert(0,"segment_ID",segment_ids[i])
            median[i] = ft_pt_median.T

        self.samples = samples.copy()
        #### If you want to save the data when it changes between patients just move it inside the loop, it will work
        if save:
            print("Saving in: ",self.filename_save)
            self.save_h5(signal_dict,mean,median,fiducials_names=fp_col, filename=self.filename_save)

        return mean,median,signal_dict
