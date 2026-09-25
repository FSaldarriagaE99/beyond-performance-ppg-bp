import pandas as pd

import pyPPG

from lib_changes.features import get_ppg_features
from pyPPG.ppg_bm.statistics import get_statistics

class BmCollection2:

    ###########################################################################
    ######################## Initialization of Biomarkers #####################
    ###########################################################################
    def __init__(self, s: pyPPG.PPG, fp: pyPPG.Fiducials):
        """
        The purpose of the Biomarkers class is to calculate the PPG biomarkers.

        :param s: object of PPG signal
        :type s: pyPPG.PPG object
        :param fp: object of fiducial points
        :type fp: pyPPG.Fiducials object

        """

        self.s = s
        self.fp = fp

    ###########################################################################
    ############################ Get PPG Biomarkers ###########################
    ###########################################################################
    def get_biomarkers2 (self, get_stat: bool):
        """
        This function retrieves the list of biomarkers, computes their values, and calculates associated statistics.

        :param get_stat: a bool for calculating the statistics of biomarkers
        :type get_stat: bool

        :return:
            - bm_defs: dictionary of biomarkers with name, definition and unit
            - bm_vals: dictionary of biomarkers with values
            - bm_stats: dictionary of biomarkers with statistics
        """

        s=self.s
        fp = self.fp

        ## Get Biomarkers
        pw_ppg_sig, bm_ppg_sig, def_ppg_sig = get_ppg_features(s, fp)

        bm_vals={'ppg_features': bm_ppg_sig}
        bm_defs = {'ppg_features': def_ppg_sig}

        ## Get Statistics
        if get_stat:
            bm_stats = get_statistics(fp.sp, fp.on, bm_vals)
        else:
            bm_stats={'ppg_features': []}

        ## Update index names
        BM_keys = bm_vals.keys()
        for key in BM_keys:
            bm_vals[key] = bm_vals[key].rename_axis('Index of pulse')
            bm_vals[key].insert(0,'TimeStamp',pw_ppg_sig.onset)
            bm_defs[key] = bm_defs[key].rename_axis('No. biomarkers')
            if get_stat: bm_stats[key] = bm_stats[key].rename_axis('Statistics')

        if get_stat:
            return bm_defs, bm_vals, bm_stats
        else:
            return bm_defs, bm_vals
