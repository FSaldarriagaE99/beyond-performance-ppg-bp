import pyPPG

import pandas as pd
from lib_changes.bm_extraction2 import get_biomarkers

###########################################################################
####################### Get Biomarkers of PPG Signal ######################
###########################################################################
def get_ppg_features(s: pyPPG.PPG, fp: pyPPG.Fiducials):
    """
    This function returns the biomarkers of PPG signal.

    :param s: object of PPG signal
    :type s: pyPPG.PPG object
    :param fp: object of fiducial points
    :type fp: pyPPG.Fiducials object

    :return:
        - df_pw: data frame with onsets, offsets and peaks
        - df_biomarkers: dictionary of biomarkers of PPG signal
        - biomarkers_lst: list a biomarkers with name, definition and unit
    """

    biomarkers_lst = [
                    ["IPR",          "Instantaneous pulse rate, 60 / Time between systolic peaks", "[%]"],
                    ["Tsp",   "Systolic peak time, the time between the pulse onset and systolic peak", "[s]"],
                    ["TWRRF25","Time width ratio at 25% of rising branch (systolic width) to falling branch (diastolic width)","[nu]"],
                    ["TWRRF50","Time width ratio at 50% of rising branch (systolic width) to falling branch (diastolic width)","[nu]"],
                    ["Tsw25", "Systolic width, the width at 25% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw50", "Systolic width, the width at 50% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw75", "Systolic width, the width at 75% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tdw25", "Diastolic width, the width at 25% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw50", "Diastolic width, the width at 50% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw75", "Diastolic width, the width at 75% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["AUCpi", "Area under pulse interval curve, the area under the pulse wave between pulse onset and pulse offset", "[nu]"],
                    ["IPA",          "Inflection point area, the ratio of the area under diastolic curve vs. the area under systolic curve", "[nu]"],
                    ["Av-Au ratio",        "Ratio of the v-point amplitude vs. the u-point amplitude", "[%]"],
                    ["Ab-Aa ratio",        "Ratio of the b-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ac-Aa ratio",        "Ratio of the c-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ad-Aa ratio",        "Ratio of the d-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ap2-Ap1 ratio",      "Ratio of the p2-point amplitude vs. the p1-point amplitude", "[%]"],
                    ["AGI",          "Aging Index, (Ab-Ac-Ad-Ae)/Aa", "[%]"],
                    ["Kurtosis",     "The measure of sharpness of the peak of distribution curve", "[nu]"],
                    ["Skewness",     "The measure of the lack of symmetry from the mean of the dataset.", "[nu]"],
                    ["L-H ratio",          "Ratio of the height of a pulse compared to its width (lenght)", "[%]"],
                    ["ShannonEntropy", "Quantify the probability density function of the distribution of changes in the PPG waveform", "[nu]"],
                    ["Tpp",   "Peak-to-peak interval, the time between two consecutive systolic peaks", "[s]"]
    ]

    header = ['name', 'definition', 'unit']
    biomarkers_lst = pd.DataFrame(biomarkers_lst, columns=header)

    df_pw, df_biomarkers = get_biomarkers(s, fp, biomarkers_lst.name)

    return df_pw, df_biomarkers, biomarkers_lst