import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotmap import DotMap

from pyPPG import PPG, Fiducials, Biomarkers
from lib_changes import PPG2, Fiducials2, Biomarkers2 ##Modified ones (Fiducials and Biomarkers are the same)

from pyPPG.datahandling import load_data, plot_fiducials, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from lib_changes import fiducials2 as FP2 ##Modified one

import pyPPG.biomarkers as BM
from lib_changes import biomarkers2 as BM2 ##Modified one

import pyPPG.ppg_sqi as SQI

class Others:
    def __init__(self, data: dict, demo_info: dict, segments_ids: dict):
        self.data = data
        self.demo_info = demo_info
        self.segments_ids = segments_ids

    def show_all_data(self, delay: int, save_fig: bool):
        data = self.data
        demo_info = self.demo_info
        ########################### Showing the data (only the PPG signal) ###########################
        for key in data.keys():
            sigs = np.arange(len(data[key]))
            print("paciente ",key)
            for s in sigs:
                x = data[key][s]
            
                fs = int(demo_info[key]["SamplingFrequency"])
                t = np.arange(0,len(x))/fs
                
                plt.figure(figsize=(10, 4))
                plt.plot(t, x)
                plt.title(f'Patient {key}, fragment {s+1}:')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitud')
                plt.grid(True)
                plt.tight_layout()
                if save_fig:
                    plt.savefig(f'signals_all/patient_{key}/signal_{s+1}.png', dpi=300)
                plt.show(block=False)
                plt.pause(delay)
                plt.close()

    def show_bypatient(self, patient_id: str, delay: int, specific_signal: bool = False):
        ################################ Showing the data for a specific patient ###########################
        data = self.data
        sigs = np.arange(len(data[patient_id]))
        print("Number of signals: ",len(sigs))
        if specific_signal:
            s  = int(input(f"Which signal do you want to see? (0 to {len(sigs)-1}): "))
            x = data[patient_id][s]
            t = np.arange(0,len(x))/125
            plt.figure(figsize=(10, 4))
            plt.plot(t, x)
            plt.title(f'Patient {patient_id}, fragment {s+1}:')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitud')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("All signals")
            for s in sigs:
                x = data[patient_id][s]
                t = np.arange(0,len(x))/125
                plt.figure(figsize=(10, 4))
                plt.plot(t, x)
                plt.title(f'Patient {patient_id}, fragment {s+1}:')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitud')
                plt.grid(True)
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(delay)
                plt.close()
    
    def signal_analysisPPG(self, patient_id: str, specific_signal: int, delay: int = 5, saving: bool = False, savingfolder: str = "saves"):
        ############################# pyPPG analysis ###########################
        data = self.data
        segment_ids = self.segments_ids
        demo_info = self.demo_info
        idx = list(segment_ids[patient_id]).index(specific_signal)

        print(f"Patient {patient_id} - signal {segment_ids[patient_id][idx]}")

        signal = DotMap()
        signal.name = patient_id
        signal.start_sig = 0 
        signal.end_sig = len(data[patient_id][idx])
        signal.v = data[patient_id][idx]
        signal.fs = int(demo_info[patient_id]["SamplingFrequency"])

        signal.filtering = True # whether or not to filter the PPG signal
        signal.fL=0.5000001 # Lower cutoff frequency (Hz)
        signal.fH=12 # Upper cutoff frequency (Hz)
        signal.order=4 # Filter order
        signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"
        prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
        signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

        # setup figure
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex = True, sharey = False)
        t = np.arange(0,len(signal.ppg))/signal.fs
        # plot filtered PPG signal
        ax1.plot(t, signal.ppg)
        ax1.set(xlabel = '', ylabel = 'PPG')
        # plot first derivative
        ax2.plot(t, signal.vpg)
        ax2.set(xlabel = '', ylabel = 'PPG\'')
        # plot second derivative
        ax3.plot(t, signal.apg)
        ax3.set(xlabel = '', ylabel = 'PPG\'\'')
        # plot third derivative
        ax4.plot(t, signal.jpg)
        ax4.set(xlabel = 'Time (s)', ylabel = 'PPG\'\'\'')
        # show plot
        plt.show()

        # Initialise the correction for fiducial points
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction=pd.DataFrame()
        correction.loc[0, corr_on] = True
        signal.correction=correction

        # Create a PPG class
        s = PPG2(signal)
        fpex = FP2.FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s)
        print("Fiducial points:\n",fiducials + s.start_sig) # here the starting sample is added so that the results are relative to the start of the original signal (rather than the start of the analysed segment)
        # Create a fiducials class
        fp = Fiducials(fp=fiducials)

        # Plot fiducial points
        plot_fiducials(s, fp, savingfolder, legend_fontsize=12, show_fig= True)

        #Estimate Heart Rate
        num_beats=len(fp.sp)  # number of the beats
        duration_seconds=len(s.ppg)/s.fs  # duration in seconds
        HR = (num_beats / duration_seconds) * 60 # heart rate
        print('Estimated HR: ',HR,' bpm' )

        # Get PPG SQI
        ppgSQI = round(np.mean(SQI.get_ppgSQI(s.ppg, s.fs, fp.sp)) * 100, 2)
        print('Mean PPG SQI: ', ppgSQI, '%')
        # Init the biomarkers package
        bmex = BM2.BmCollection2(s=s, fp=fp)

        # Extract biomarkers
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers2(get_stat=True)
        tmp_keys=bm_stats.keys()
        print('Statistics of the biomarkers:')
        for i in tmp_keys: print(i,'\n',bm_stats[i])

        # Create a biomarkers class
        bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)
        # Save PPG struct, fiducial points, biomarkers
        fp_new = Fiducials(fp.get_fp() + s.start_sig) # here the starting sample is added so that the results are relative to the start of the original signal (rather than the start of the analysed segment)
        if saving:
            save_data(s=s, fp=fp_new, bm=bm, savingformat="csv", savingfolder=savingfolder)
        print("bm_vals",bm_defs)
        print(num_beats)
    