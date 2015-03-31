# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from functools import wraps
import time


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("trex2_model_rest.py@timefn: " + fn.func_name + " took " + "{:.6f}".format(t2 - t1) + " seconds")
        return result

    return measure_time


class trex2(object):
    @timefn
    def __init__(self, run_type, pd_obj, pd_obj_exp):
        logging.info("************** trex back end **********************")
        # run_type can be single, batch or qaqc
        self.run_type = run_type

        # Inputs: Assign object attribute variables from the input Pandas DataFrame
        self.chem_name = pd_obj['chem_name']
        self.use = pd_obj['use']
        self.formu_name = pd_obj['formu_name']
        self.a_i = pd_obj['a_i']
        self.a_i /= 100  # change from percentage to proportion
        self.Application_type = pd_obj['Application_type']
        self.seed_treatment_formulation_name = pd_obj['seed_treatment_formulation_name']
        self.seed_crop = pd_obj['seed_crop']
        self.seed_crop_v = pd_obj['seed_crop_v']
        self.r_s = pd_obj['r_s']
        self.b_w = pd_obj['b_w']
        self.b_w /= 12  # convert to ft
        self.p_i = pd_obj['p_i']
        self.p_i /= 100  # change from percentage to proportion
        self.den = pd_obj['den']
        self.h_l = pd_obj['h_l']  # half-life
        self.noa = pd_obj['noa']  # number of applications

        self.rate_out = []  # application rates for each application day (needs to be built)
        self.day_out = []  # day numbers of the applications (needs to be built as a dataframe)
        logging.info(range(self.noa.iloc[0]))
        napps = self.noa.iloc[0]
        for i in range(napps):
            logging.info(i)
            # rate_temp = request.POST.get('rate'+str(j))
            rate_temp = getattr(pd_obj, 'rate' + str(i))
            self.rate_out.append(float(rate_temp))
            logging.info("self.rate_out")
            logging.info(self.rate_out)
            # #day_temp = float(request.POST.get('day'+str(j)))
            day_temp = getattr(pd_obj, 'day' + str(i))
            self.day_out.append(day_temp)
            logging.info("self.day_out")
            logging.info(self.day_out)

        # self.ar_lb = self.rate_out[0] #?
        self.first_app_lb = pd.Series(name="first_app_lb")
        self.first_app_lb = self.rate_out[0]

        self.ld50_bird = pd_obj['ld50_bird']
        self.lc50_bird = pd_obj['lc50_bird']
        self.NOAEC_bird = pd_obj['NOAEC_bird']
        self.NOAEL_bird = pd_obj['NOAEL_bird']
        self.aw_bird_sm = pd_obj['aw_bird_sm']
        self.aw_bird_md = pd_obj['aw_bird_md']
        self.aw_bird_lg = pd_obj['aw_bird_lg']

        self.Species_of_the_tested_bird_avian_ld50 = pd_obj['Species_of_the_tested_bird_avian_ld50']
        self.Species_of_the_tested_bird_avian_lc50 = pd_obj['Species_of_the_tested_bird_avian_lc50']
        self.Species_of_the_tested_bird_avian_NOAEC = pd_obj['Species_of_the_tested_bird_avian_NOAEC']
        self.Species_of_the_tested_bird_avian_NOAEL = pd_obj['Species_of_the_tested_bird_avian_NOAEL']

        self.tw_bird_ld50 = pd_obj['tw_bird_ld50']
        self.tw_bird_lc50 = pd_obj['tw_bird_lc50']
        self.tw_bird_NOAEC = pd_obj['tw_bird_NOAEC']
        self.tw_bird_NOAEL = pd_obj['tw_bird_NOAEL']
        self.x = pd_obj['x']  # mineau scaling factor
        self.ld50_mamm = pd_obj['ld50_mamm']
        self.lc50_mamm = pd_obj['lc50_mamm']
        self.NOAEC_mamm = pd_obj['NOAEC_mamm']
        self.NOAEL_mamm = pd_obj['NOAEL_mamm']
        self.aw_mamm_sm = pd_obj['aw_mamm_sm']
        self.aw_mamm_md = pd_obj['aw_mamm_md']
        self.aw_mamm_lg = pd_obj['aw_mamm_lg']
        self.tw_mamm = pd_obj['tw_mamm']
        self.m_s_r_p = pd_obj['m_s_r_p']

        # Outputs: Assign object attribute variables to Pandas Series
        # initial concentrations for different food types
        self.C_0_sg = pd.Series(name="C_0_sg")  # short grass
        self.C_0_tg = pd.Series(name="C_0_tg")  # tall grass
        self.C_0_blp = pd.Series(name="C_0_blp")  # broad-leafed plants
        self.C_0_fp = pd.Series(name="C_0_fp")  # fruits/pods
        self.C_0_arthro = pd.Series(name="C_0_arthro")  # arthropods

        # mean concentration estimate based on first application rate
        self.C_mean_sg = pd.Series(name="C_mean_sg")  # short grass
        self.C_mean_tg = pd.Series(name="C_mean_tg")  # tall grass
        self.C_mean_blp = pd.Series(name="C_mean_blp")  # broad-leafed plants
        self.C_mean_fp = pd.Series(name="C_mean_fp")  # fruits/pods
        self.C_mean_arthro = pd.Series(name="C_mean_arthro")  # arthropods

        # time series estimate based on first application rate - needs to be matrices for batch runs
        self.C_ts_sg = pd.Series(name="C_ts_sg")  # short grass
        self.C_ts_tg = pd.Series(name="C_ts_tg")  # tall grass
        self.C_ts_blp = pd.Series(name="C_ts_blp")  # broad-leafed plants
        self.C_ts_fp = pd.Series(name="C_ts_fp")  # fruits/pods
        self.C_ts_arthro = pd.Series(name="C_ts_arthro")  # arthropods

        # Table5
        self.sa_bird_1_s = pd.Series(name="sa_bird_1_s")
        self.sa_bird_2_s = pd.Series(name="sa_bird_2_s")
        self.sc_bird_s = pd.Series(name="sc_bird_s")
        self.sa_mamm_1_s = pd.Series(name="sa_mamm_1_s")
        self.sa_mamm_2_s = pd.Series(name="sa_mamm_2_s")
        self.sc_mamm_s = pd.Series(name="sc_mamm_s")

        self.sa_bird_1_m = pd.Series(name="sa_bird_1_m")
        self.sa_bird_2_m = pd.Series(name="sa_bird_2_m")
        self.sc_bird_m = pd.Series(name="sc_bird_m")
        self.sa_mamm_1_m = pd.Series(name="sa_mamm_1_m")
        self.sa_mamm_2_m = pd.Series(name="sa_mamm_2_m")
        self.sc_mamm_m = pd.Series(name="sc_mamm_m")

        self.sa_bird_1_l = pd.Series(name="sa_bird_1_l")
        self.sa_bird_2_l = pd.Series(name="sa_bird_2_l")
        self.sc_bird_l = pd.Series(name="sc_bird_l")
        self.sa_mamm_1_l = pd.Series(name="sa_mamm_1_l")
        self.sa_mamm_2_l = pd.Series(name="sa_mamm_2_l")
        self.sc_mamm_l = pd.Series(name="sc_mamm_l")

        # Table 6
        self.EEC_diet_SG = pd.Series(name="EEC_diet_SG")
        self.EEC_diet_TG = pd.Series(name="EEC_diet_TG")
        self.EEC_diet_BP = pd.Series(name="EEC_diet_BP")
        self.EEC_diet_FR = pd.Series(name="EEC_diet_FR")
        self.EEC_diet_AR = pd.Series(name="EEC_diet_AR")

        # Table 7
        self.EEC_dose_bird_SG_sm = pd.Series(name="EEC_dose_bird_SG_sm")
        self.EEC_dose_bird_SG_md = pd.Series(name="EEC_dose_bird_SG_md")
        self.EEC_dose_bird_SG_lg = pd.Series(name="EEC_dose_bird_SG_lg")
        self.EEC_dose_bird_TG_sm = pd.Series(name="EEC_dose_bird_TG_sm")
        self.EEC_dose_bird_TG_md = pd.Series(name="EEC_dose_bird_TG_md")
        self.EEC_dose_bird_TG_lg = pd.Series(name="EEC_dose_bird_TG_lg")
        self.EEC_dose_bird_BP_sm = pd.Series(name="EEC_dose_bird_BP_sm")
        self.EEC_dose_bird_BP_md = pd.Series(name="EEC_dose_bird_BP_md")
        self.EEC_dose_bird_BP_lg = pd.Series(name="EEC_dose_bird_BP_lg")
        self.EEC_dose_bird_FP_sm = pd.Series(name="EEC_dose_bird_FP_sm")
        self.EEC_dose_bird_FP_md = pd.Series(name="EEC_dose_bird_FP_md")
        self.EEC_dose_bird_FP_lg = pd.Series(name="EEC_dose_bird_FP_lg")
        self.EEC_dose_bird_AR_sm = pd.Series(name="EEC_dose_bird_AR_sm")
        self.EEC_dose_bird_AR_md = pd.Series(name="EEC_dose_bird_AR_md")
        self.EEC_dose_bird_AR_lg = pd.Series(name="EEC_dose_bird_AR_lg")
        self.EEC_dose_bird_SE_sm = pd.Series(name="EEC_dose_bird_SE_sm")
        self.EEC_dose_bird_SE_md = pd.Series(name="EEC_dose_bird_SE_md")
        self.EEC_dose_bird_SE_lg = pd.Series(name="EEC_dose_bird_SE_lg")

        # Table 7_add
        self.ARQ_bird_SG_sm = pd.Series(name="ARQ_bird_SG_sm")
        self.ARQ_bird_SG_md = pd.Series(name="ARQ_bird_SG_md")
        self.ARQ_bird_SG_lg = pd.Series(name="ARQ_bird_SG_lg")
        self.ARQ_bird_TG_sm = pd.Series(name="ARQ_bird_TG_sm")
        self.ARQ_bird_TG_md = pd.Series(name="ARQ_bird_TG_md")
        self.ARQ_bird_TG_lg = pd.Series(name="ARQ_bird_TG_lg")
        self.ARQ_bird_BP_sm = pd.Series(name="ARQ_bird_BP_sm")
        self.ARQ_bird_BP_md = pd.Series(name="ARQ_bird_BP_md")
        self.ARQ_bird_BP_lg = pd.Series(name="ARQ_bird_BP_lg")
        self.ARQ_bird_FP_sm = pd.Series(name="ARQ_bird_FP_sm")
        self.ARQ_bird_FP_md = pd.Series(name="ARQ_bird_FP_md")
        self.ARQ_bird_FP_lg = pd.Series(name="ARQ_bird_FP_lg")
        self.ARQ_bird_AR_sm = pd.Series(name="ARQ_bird_AR_sm")
        self.ARQ_bird_AR_md = pd.Series(name="ARQ_bird_AR_md")
        self.ARQ_bird_AR_lg = pd.Series(name="ARQ_bird_AR_lg")
        self.ARQ_bird_SE_sm = pd.Series(name="ARQ_bird_SE_sm")
        self.ARQ_bird_SE_md = pd.Series(name="ARQ_bird_SE_md")
        self.ARQ_bird_SE_lg = pd.Series(name="ARQ_bird_SE_lg")

        # Table 8
        self.ARQ_diet_bird_SG_A = pd.Series(name="ARQ_diet_bird_SG_A")
        self.ARQ_diet_bird_SG_C = pd.Series(name="ARQ_diet_bird_SG_C")
        self.ARQ_diet_bird_TG_A = pd.Series(name="ARQ_diet_bird_TG_A")
        self.ARQ_diet_bird_TG_C = pd.Series(name="ARQ_diet_bird_TG_C")
        self.ARQ_diet_bird_BP_A = pd.Series(name="ARQ_diet_bird_BP_A")
        self.ARQ_diet_bird_BP_C = pd.Series(name="ARQ_diet_bird_BP_C")
        self.ARQ_diet_bird_FP_A = pd.Series(name="ARQ_diet_bird_FP_A")
        self.ARQ_diet_bird_FP_C = pd.Series(name="ARQ_diet_bird_FP_C")
        self.ARQ_diet_bird_AR_A = pd.Series(name="ARQ_diet_bird_AR_A")
        self.ARQ_diet_bird_AR_C = pd.Series(name="ARQ_diet_bird_AR_C")

        # Table 9
        self.EEC_dose_mamm_SG_sm = pd.Series(name="EEC_dose_mamm_SG_sm")
        self.EEC_dose_mamm_SG_md = pd.Series(name="EEC_dose_mamm_SG_md")
        self.EEC_dose_mamm_SG_lg = pd.Series(name="EEC_dose_mamm_SG_lg")
        self.EEC_dose_mamm_TG_sm = pd.Series(name="EEC_dose_mamm_TG_sm")
        self.EEC_dose_mamm_TG_md = pd.Series(name="EEC_dose_mamm_TG_md")
        self.EEC_dose_mamm_TG_lg = pd.Series(name="EEC_dose_mamm_TG_lg")
        self.EEC_dose_mamm_BP_sm = pd.Series(name="EEC_dose_mamm_BP_sm")
        self.EEC_dose_mamm_BP_md = pd.Series(name="EEC_dose_mamm_BP_md")
        self.EEC_dose_mamm_BP_lg = pd.Series(name="EEC_dose_mamm_BP_lg")
        self.EEC_dose_mamm_FP_sm = pd.Series(name="EEC_dose_mamm_FP_sm")
        self.EEC_dose_mamm_FP_md = pd.Series(name="EEC_dose_mamm_FP_md")
        self.EEC_dose_mamm_FP_lg = pd.Series(name="EEC_dose_mamm_FP_lg")
        self.EEC_dose_mamm_AR_sm = pd.Series(name="EEC_dose_mamm_AR_sm")
        self.EEC_dose_mamm_AR_md = pd.Series(name="EEC_dose_mamm_AR_md")
        self.EEC_dose_mamm_AR_lg = pd.Series(name="EEC_dose_mamm_AR_lg")
        self.EEC_dose_mamm_SE_sm = pd.Series(name="EEC_dose_mamm_SE_sm")
        self.EEC_dose_mamm_SE_md = pd.Series(name="EEC_dose_mamm_SE_md")
        self.EEC_dose_mamm_SE_lg = pd.Series(name="EEC_dose_mamm_SE_lg")

        # Table 10
        self.ARQ_dose_mamm_SG_sm = pd.Series(name="ARQ_dose_mamm_SG_sm")
        self.CRQ_dose_mamm_SG_sm = pd.Series(name="CRQ_dose_mamm_SG_sm")
        self.ARQ_dose_mamm_SG_md = pd.Series(name="ARQ_dose_mamm_SG_md")
        self.CRQ_dose_mamm_SG_md = pd.Series(name="CRQ_dose_mamm_SG_md")
        self.ARQ_dose_mamm_SG_lg = pd.Series(name="ARQ_dose_mamm_SG_lg")
        self.CRQ_dose_mamm_SG_lg = pd.Series(name="CRQ_dose_mamm_SG_lg")

        self.ARQ_dose_mamm_TG_sm = pd.Series(name="ARQ_dose_mamm_TG_sm")
        self.CRQ_dose_mamm_TG_sm = pd.Series(name="CRQ_dose_mamm_TG_sm")
        self.ARQ_dose_mamm_TG_md = pd.Series(name="ARQ_dose_mamm_TG_md")
        self.CRQ_dose_mamm_TG_md = pd.Series(name="CRQ_dose_mamm_TG_md")
        self.ARQ_dose_mamm_TG_lg = pd.Series(name="ARQ_dose_mamm_TG_lg")
        self.CRQ_dose_mamm_TG_lg = pd.Series(name="CRQ_dose_mamm_TG_lg")

        self.ARQ_dose_mamm_BP_sm = pd.Series(name="ARQ_dose_mamm_BP_sm")
        self.CRQ_dose_mamm_BP_sm = pd.Series(name="CRQ_dose_mamm_BP_sm")
        self.ARQ_dose_mamm_BP_md = pd.Series(name="ARQ_dose_mamm_BP_md")
        self.CRQ_dose_mamm_BP_md = pd.Series(name="CRQ_dose_mamm_BP_md")
        self.ARQ_dose_mamm_BP_lg = pd.Series(name="ARQ_dose_mamm_BP_lg")
        self.CRQ_dose_mamm_BP_lg = pd.Series(name="CRQ_dose_mamm_BP_lg")

        self.ARQ_dose_mamm_FP_sm = pd.Series(name="ARQ_dose_mamm_FP_sm")
        self.CRQ_dose_mamm_FP_sm = pd.Series(name="CRQ_dose_mamm_FP_sm")
        self.ARQ_dose_mamm_FP_md = pd.Series(name="ARQ_dose_mamm_FP_md")
        self.CRQ_dose_mamm_FP_md = pd.Series(name="CRQ_dose_mamm_FP_md")
        self.ARQ_dose_mamm_FP_lg = pd.Series(name="ARQ_dose_mamm_FP_lg")
        self.CRQ_dose_mamm_FP_lg = pd.Series(name="CRQ_dose_mamm_FP_lg")

        self.ARQ_dose_mamm_AR_sm = pd.Series(name="ARQ_dose_mamm_AR_sm")
        self.CRQ_dose_mamm_AR_sm = pd.Series(name="CRQ_dose_mamm_AR_sm")
        self.ARQ_dose_mamm_AR_md = pd.Series(name="ARQ_dose_mamm_AR_md")
        self.CRQ_dose_mamm_AR_md = pd.Series(name="CRQ_dose_mamm_AR_md")
        self.ARQ_dose_mamm_AR_lg = pd.Series(name="ARQ_dose_mamm_AR_lg")
        self.CRQ_dose_mamm_AR_lg = pd.Series(name="CRQ_dose_mamm_AR_lg")

        self.ARQ_dose_mamm_SE_sm = pd.Series(name="ARQ_dose_mamm_SE_sm")
        self.CRQ_dose_mamm_SE_sm = pd.Series(name="CRQ_dose_mamm_SE_sm")
        self.ARQ_dose_mamm_SE_md = pd.Series(name="ARQ_dose_mamm_SE_md")
        self.CRQ_dose_mamm_SE_md = pd.Series(name="CRQ_dose_mamm_SE_md")
        self.ARQ_dose_mamm_SE_lg = pd.Series(name="ARQ_dose_mamm_SE_lg")
        self.CRQ_dose_mamm_SE_lg = pd.Series(name="CRQ_dose_mamm_SE_lg")

        # table 11
        self.ARQ_diet_mamm_SG = pd.Series(name="ARQ_diet_mamm_SG")
        self.ARQ_diet_mamm_TG = pd.Series(name="ARQ_diet_mamm_TG")
        self.ARQ_diet_mamm_BP = pd.Series(name="ARQ_diet_mamm_BP")
        self.ARQ_diet_mamm_FP = pd.Series(name="ARQ_diet_mamm_FP")
        self.ARQ_diet_mamm_AR = pd.Series(name="ARQ_diet_mamm_AR")

        self.CRQ_diet_mamm_SG = pd.Series(name="CRQ_diet_mamm_SG")
        self.CRQ_diet_mamm_TG = pd.Series(name="CRQ_diet_mamm_TG")
        self.CRQ_diet_mamm_BP = pd.Series(name="CRQ_diet_mamm_BP")
        self.CRQ_diet_mamm_FP = pd.Series(name="CRQ_diet_mamm_FP")
        self.CRQ_diet_mamm_AR = pd.Series(name="CRQ_diet_mamm_AR")

        # Table12
        self.LD50_rg_bird_sm = pd.Series(name="LD50_rg_bird_sm")
        self.LD50_rg_mamm_sm = pd.Series(name="LD50_rg_mamm_sm")
        self.LD50_rg_bird_md = pd.Series(name="LD50_rg_bird_md")
        self.LD50_rg_mamm_md = pd.Series(name="LD50_rg_mamm_md")
        self.LD50_rg_bird_lg = pd.Series(name="LD50_rg_bird_lg")
        self.LD50_rg_mamm_lg = pd.Series(name="LD50_rg_mamm_lg")

        # Table13
        self.LD50_rl_bird_sm = pd.Series(name="LD50_rl_bird_sm")
        self.LD50_rl_mamm_sm = pd.Series(name="LD50_rl_mamm_sm")
        self.LD50_rl_bird_md = pd.Series(name="LD50_rl_bird_md")
        self.LD50_rl_mamm_md = pd.Series(name="LD50_rl_mamm_md")
        self.LD50_rl_bird_lg = pd.Series(name="LD50_rl_bird_lg")
        self.LD50_rl_mamm_lg = pd.Series(name="LD50_rl_mamm_lg")

        # Table14
        self.LD50_bg_bird_sm = pd.Series(name="LD50_bg_bird_sm")
        self.LD50_bg_mamm_sm = pd.Series(name="LD50_bg_mamm_sm")
        self.LD50_bg_bird_md = pd.Series(name="LD50_bg_bird_md")
        self.LD50_bg_mamm_md = pd.Series(name="LD50_bg_mamm_md")
        self.LD50_bg_bird_lg = pd.Series(name="LD50_bg_bird_lg")
        self.LD50_bg_mamm_lg = pd.Series(name="LD50_bg_mamm_lg")

        # Table15
        self.LD50_bl_bird_sm = pd.Series(name="LD50_bl_bird_sm")
        self.LD50_bl_mamm_sm = pd.Series(name="LD50_bl_mamm_sm")
        self.LD50_bl_bird_md = pd.Series(name="LD50_bl_bird_md")
        self.LD50_bl_mamm_md = pd.Series(name="LD50_bl_mamm_md")
        self.LD50_bl_bird_lg = pd.Series(name="LD50_bl_bird_lg")
        self.LD50_bl_mamm_lg = pd.Series(name="LD50_bl_mamm_lg")

        # Execute model methods
        self.run_methods()

        # Create DataFrame containing output value Series
        pd_obj_out = pd.DataFrame({
            # Table5
            'sa_bird_1_s': self.sa_bird_1_s,
            'sa_bird_2_s': self.sa_bird_2_s,
            'sc_bird_s': self.sc_bird_s,
            'sa_mamm_1_s': self.sa_mamm_1_s,
            'sa_mamm_2_s': self.sa_mamm_2_s,
            'sc_mamm_s': self.sc_mamm_s,

            'sa_bird_1_m': self.sa_bird_1_m,
            'sa_bird_2_m': self.sa_bird_2_m,
            'sc_bird_m': self.sc_bird_m,
            'sa_mamm_1_m': self.sa_mamm_1_m,
            'sa_mamm_2_m': self.sa_mamm_2_m,
            'sc_mamm_m': self.sc_mamm_m,

            'sa_bird_1_l': self.sa_bird_1_l,
            'sa_bird_2_l': self.sa_bird_2_l,
            'sc_bird_l': self.sc_bird_l,
            'sa_mamm_1_l': self.sa_mamm_1_l,
            'sa_mamm_2_l': self.sa_mamm_2_l,
            'sc_mamm_l': self.sc_mamm_l,

            # Table 6
            'EEC_diet_SG': self.EEC_diet_SG,
            'EEC_diet_TG': self.EEC_diet_TG,
            'EEC_diet_BP': self.EEC_diet_BP,
            'EEC_diet_FR': self.EEC_diet_FR,
            'EEC_diet_AR': self.EEC_diet_AR,

            # Table 7
            'EEC_dose_bird_SG_sm': self.EEC_dose_bird_SG_sm,
            'EEC_dose_bird_SG_md': self.EEC_dose_bird_SG_md,
            'EEC_dose_bird_SG_lg': self.EEC_dose_bird_SG_lg,
            'EEC_dose_bird_TG_sm': self.EEC_dose_bird_TG_sm,
            'EEC_dose_bird_TG_md': self.EEC_dose_bird_TG_md,
            'EEC_dose_bird_TG_lg': self.EEC_dose_bird_TG_lg,
            'EEC_dose_bird_BP_sm': self.EEC_dose_bird_BP_sm,
            'EEC_dose_bird_BP_md': self.EEC_dose_bird_BP_md,
            'EEC_dose_bird_BP_lg': self.EEC_dose_bird_BP_lg,
            'EEC_dose_bird_FP_sm': self.EEC_dose_bird_FP_sm,
            'EEC_dose_bird_FP_md': self.EEC_dose_bird_FP_md,
            'EEC_dose_bird_FP_lg': self.EEC_dose_bird_FP_lg,
            'EEC_dose_bird_AR_sm': self.EEC_dose_bird_AR_sm,
            'EEC_dose_bird_AR_md': self.EEC_dose_bird_AR_md,
            'EEC_dose_bird_AR_lg': self.EEC_dose_bird_AR_lg,
            'EEC_dose_bird_SE_sm': self.EEC_dose_bird_SE_sm,
            'EEC_dose_bird_SE_md': self.EEC_dose_bird_SE_md,
            'EEC_dose_bird_SE_lg': self.EEC_dose_bird_SE_lg,

            # Table 7_add
            'ARQ_bird_SG_sm': self.ARQ_bird_SG_sm,
            'ARQ_bird_SG_md': self.ARQ_bird_SG_md,
            'ARQ_bird_SG_lg': self.ARQ_bird_SG_lg,
            'ARQ_bird_TG_sm': self.ARQ_bird_TG_sm,
            'ARQ_bird_TG_md': self.ARQ_bird_TG_md,
            'ARQ_bird_TG_lg': self.ARQ_bird_TG_lg,
            'ARQ_bird_BP_sm': self.ARQ_bird_BP_sm,
            'ARQ_bird_BP_md': self.ARQ_bird_BP_md,
            'ARQ_bird_BP_lg': self.ARQ_bird_BP_lg,
            'ARQ_bird_FP_sm': self.ARQ_bird_FP_sm,
            'ARQ_bird_FP_md': self.ARQ_bird_FP_md,
            'ARQ_bird_FP_lg': self.ARQ_bird_FP_lg,
            'ARQ_bird_AR_sm': self.ARQ_bird_AR_sm,
            'ARQ_bird_AR_md': self.ARQ_bird_AR_md,
            'ARQ_bird_AR_lg': self.ARQ_bird_AR_lg,
            'ARQ_bird_SE_sm': self.ARQ_bird_SE_sm,
            'ARQ_bird_SE_md': self.ARQ_bird_SE_md,
            'ARQ_bird_SE_lg': self.ARQ_bird_SE_lg,

            # Table 8
            'ARQ_diet_bird_SG_A': self.ARQ_diet_bird_SG_A,
            'ARQ_diet_bird_SG_C': self.ARQ_diet_bird_SG_C,
            'ARQ_diet_bird_TG_A': self.ARQ_diet_bird_TG_A,
            'ARQ_diet_bird_TG_C': self.ARQ_diet_bird_TG_C,
            'ARQ_diet_bird_BP_A': self.ARQ_diet_bird_BP_A,
            'ARQ_diet_bird_BP_C': self.ARQ_diet_bird_BP_C,
            'ARQ_diet_bird_FP_A': self.ARQ_diet_bird_FP_A,
            'ARQ_diet_bird_FP_C': self.ARQ_diet_bird_FP_C,
            'ARQ_diet_bird_AR_A': self.ARQ_diet_bird_AR_A,
            'ARQ_diet_bird_AR_C': self.ARQ_diet_bird_AR_C,

            # Table 9
            'EEC_dose_mamm_SG_sm': self.EEC_dose_mamm_SG_sm,
            'EEC_dose_mamm_SG_md': self.EEC_dose_mamm_SG_md,
            'EEC_dose_mamm_SG_lg': self.EEC_dose_mamm_SG_lg,
            'EEC_dose_mamm_TG_sm': self.EEC_dose_mamm_TG_sm,
            'EEC_dose_mamm_TG_md': self.EEC_dose_mamm_TG_md,
            'EEC_dose_mamm_TG_lg': self.EEC_dose_mamm_TG_lg,
            'EEC_dose_mamm_BP_sm': self.EEC_dose_mamm_BP_sm,
            'EEC_dose_mamm_BP_md': self.EEC_dose_mamm_BP_md,
            'EEC_dose_mamm_BP_lg': self.EEC_dose_mamm_BP_lg,
            'EEC_dose_mamm_FP_sm': self.EEC_dose_mamm_FP_sm,
            'EEC_dose_mamm_FP_md': self.EEC_dose_mamm_FP_md,
            'EEC_dose_mamm_FP_lg': self.EEC_dose_mamm_FP_lg,
            'EEC_dose_mamm_AR_sm': self.EEC_dose_mamm_AR_sm,
            'EEC_dose_mamm_AR_md': self.EEC_dose_mamm_AR_md,
            'EEC_dose_mamm_AR_lg': self.EEC_dose_mamm_AR_lg,
            'EEC_dose_mamm_SE_sm': self.EEC_dose_mamm_SE_sm,
            'EEC_dose_mamm_SE_md': self.EEC_dose_mamm_SE_md,
            'EEC_dose_mamm_SE_lg': self.EEC_dose_mamm_SE_lg,

            # Table 10
            'ARQ_dose_mamm_SG_sm': self.ARQ_dose_mamm_SG_sm,
            'CRQ_dose_mamm_SG_sm': self.CRQ_dose_mamm_SG_sm,
            'ARQ_dose_mamm_SG_md': self.ARQ_dose_mamm_SG_md,
            'CRQ_dose_mamm_SG_md': self.CRQ_dose_mamm_SG_md,
            'ARQ_dose_mamm_SG_lg': self.ARQ_dose_mamm_SG_lg,
            'CRQ_dose_mamm_SG_lg': self.CRQ_dose_mamm_SG_lg,

            'ARQ_dose_mamm_TG_sm': self.ARQ_dose_mamm_TG_sm,
            'CRQ_dose_mamm_TG_sm': self.CRQ_dose_mamm_TG_sm,
            'ARQ_dose_mamm_TG_md': self.ARQ_dose_mamm_TG_md,
            'CRQ_dose_mamm_TG_md': self.CRQ_dose_mamm_TG_md,
            'ARQ_dose_mamm_TG_lg': self.ARQ_dose_mamm_TG_lg,
            'CRQ_dose_mamm_TG_lg': self.CRQ_dose_mamm_TG_lg,

            'ARQ_dose_mamm_BP_sm': self.ARQ_dose_mamm_BP_sm,
            'CRQ_dose_mamm_BP_sm': self.CRQ_dose_mamm_BP_sm,
            'ARQ_dose_mamm_BP_md': self.ARQ_dose_mamm_BP_md,
            'CRQ_dose_mamm_BP_md': self.CRQ_dose_mamm_BP_md,
            'ARQ_dose_mamm_BP_lg': self.ARQ_dose_mamm_BP_lg,
            'CRQ_dose_mamm_BP_lg': self.CRQ_dose_mamm_BP_lg,

            'ARQ_dose_mamm_FP_sm': self.ARQ_dose_mamm_FP_sm,
            'CRQ_dose_mamm_FP_sm': self.CRQ_dose_mamm_FP_sm,
            'ARQ_dose_mamm_FP_md': self.ARQ_dose_mamm_FP_md,
            'CRQ_dose_mamm_FP_md': self.CRQ_dose_mamm_FP_md,
            'ARQ_dose_mamm_FP_lg': self.ARQ_dose_mamm_FP_lg,
            'CRQ_dose_mamm_FP_lg': self.CRQ_dose_mamm_FP_lg,

            'ARQ_dose_mamm_AR_sm': self.ARQ_dose_mamm_AR_sm,
            'CRQ_dose_mamm_AR_sm': self.CRQ_dose_mamm_AR_sm,
            'ARQ_dose_mamm_AR_md': self.ARQ_dose_mamm_AR_md,
            'CRQ_dose_mamm_AR_md': self.CRQ_dose_mamm_AR_md,
            'ARQ_dose_mamm_AR_lg': self.ARQ_dose_mamm_AR_lg,
            'CRQ_dose_mamm_AR_lg': self.CRQ_dose_mamm_AR_lg,

            'ARQ_dose_mamm_SE_sm': self.ARQ_dose_mamm_SE_sm,
            'CRQ_dose_mamm_SE_sm': self.CRQ_dose_mamm_SE_sm,
            'ARQ_dose_mamm_SE_md': self.ARQ_dose_mamm_SE_md,
            'CRQ_dose_mamm_SE_md': self.CRQ_dose_mamm_SE_md,
            'ARQ_dose_mamm_SE_lg': self.ARQ_dose_mamm_SE_lg,
            'CRQ_dose_mamm_SE_lg': self.CRQ_dose_mamm_SE_lg,

            # table 11
            'ARQ_diet_mamm_SG': self.ARQ_diet_mamm_SG,
            'ARQ_diet_mamm_TG': self.ARQ_diet_mamm_TG,
            'ARQ_diet_mamm_BP': self.ARQ_diet_mamm_BP,
            'ARQ_diet_mamm_FP': self.ARQ_diet_mamm_FP,
            'ARQ_diet_mamm_AR': self.ARQ_diet_mamm_AR,

            'CRQ_diet_mamm_SG': self.CRQ_diet_mamm_SG,
            'CRQ_diet_mamm_TG': self.CRQ_diet_mamm_TG,
            'CRQ_diet_mamm_BP': self.CRQ_diet_mamm_BP,
            'CRQ_diet_mamm_FP': self.CRQ_diet_mamm_FP,
            'CRQ_diet_mamm_AR': self.CRQ_diet_mamm_AR,

            # Table12
            'LD50_rg_bird_sm': self.LD50_rg_bird_sm,
            'LD50_rg_mamm_sm': self.LD50_rg_mamm_sm,
            'LD50_rg_bird_md': self.LD50_rg_bird_md,
            'LD50_rg_mamm_md': self.LD50_rg_mamm_md,
            'LD50_rg_bird_lg': self.LD50_rg_bird_lg,
            'LD50_rg_mamm_lg': self.LD50_rg_mamm_lg,

            # Table13
            'LD50_rl_bird_sm': self.LD50_rl_bird_sm,
            'LD50_rl_mamm_sm': self.LD50_rl_mamm_sm,
            'LD50_rl_bird_md': self.LD50_rl_bird_md,
            'LD50_rl_mamm_md': self.LD50_rl_mamm_md,
            'LD50_rl_bird_lg': self.LD50_rl_bird_lg,
            'LD50_rl_mamm_lg': self.LD50_rl_mamm_lg,

            # Table14
            'LD50_bg_bird_sm': self.LD50_bg_bird_sm,
            'LD50_bg_mamm_sm': self.LD50_bg_mamm_sm,
            'LD50_bg_bird_md': self.LD50_bg_bird_md,
            'LD50_bg_mamm_md': self.LD50_bg_mamm_md,
            'LD50_bg_bird_lg': self.LD50_bg_bird_lg,
            'LD50_bg_mamm_lg': self.LD50_bg_mamm_lg,

            # Table15
            'LD50_bl_bird_sm': self.LD50_bl_bird_sm,
            'LD50_bl_mamm_sm': self.LD50_bl_mamm_sm,
            'LD50_bl_bird_md': self.LD50_bl_bird_md,
            'LD50_bl_mamm_md': self.LD50_bl_mamm_md,
            'LD50_bl_bird_lg': self.LD50_bl_bird_lg,
            'LD50_bl_mamm_lg': self.LD50_bl_mamm_lg
        })

        # Callable from Bottle that returns JSON
        self.json = self.json(pd_obj, pd_obj_out, pd_obj_exp)

    @timefn
    def json(self, pd_obj, pd_obj_out, pd_obj_exp):
        """
            Convert DataFrames to JSON, returning a tuple 
            of JSON strings (inputs, outputs, exp_out)
        """

        pd_obj_json = pd_obj.to_json()
        pd_obj_out_json = pd_obj_out.to_json()
        try:
            pd_obj_exp_json = pd_obj_exp.to_json()
        except:
            pd_obj_exp_json = "{}"

        return pd_obj_json, pd_obj_out_json, pd_obj_exp_json

    # Begin model methods
    @timefn
    def run_methods(self):
        logging.info("run_methods")
        # build time series for each type

        # initial concentrations for different food types
        self.C_0_sg = self.first_app_lb * self.a_i * 240.  # short grass
        self.C_0_tg = self.first_app_lb * self.a_i * 110.  # tall grass
        self.C_0_blp = self.first_app_lb * self.a_i * 135.  # broad-leafed plants
        self.C_0_fp = self.first_app_lb * self.a_i * 15.  # fruits/pods
        self.C_0_arthro = self.first_app_lb * self.a_i * 94.  # arthropods

        # mean concentration estimate based on first application rate
        self.C_mean_sg = self.first_app_lb * self.a_i * 85.  # short grass
        self.C_mean_tg = self.first_app_lb * self.a_i * 36.  # tall grass
        self.C_mean_blp = self.first_app_lb * self.a_i * 45.  # broad-leafed plants
        self.C_mean_fp = self.first_app_lb * self.a_i * 7.  # fruits/pods
        self.C_mean_arthro = self.first_app_lb * self.a_i * 65.  # arthropods

        # time series estimate based on first application rate - needs to be matrices for batch runs
        self.C_ts_sg = self.C_timeseries(240)  # short grass
        self.C_ts_tg = self.C_timeseries(110)  # tall grass
        self.C_ts_blp = self.C_timeseries(135)  # broad-leafed plants
        self.C_ts_fp = self.C_timeseries(15)  # fruits/pods
        self.C_ts_arthro = self.C_timeseries(94)  # arthropods

        # Table5
        logging.info("table 5")
        self.sa_bird_1_s = self.sa_bird_1(0.1, 0.02, self.aw_bird_sm, self.tw_bird_ld50)
        self.sa_bird_2_s = self.sa_bird_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_bird,
                                          self.ld50_bird, self.aw_bird_sm, self.tw_bird_ld50, self.x, 0.02)
        self.sc_bird_s = self.sc_bird(self.first_app_lb, self.a_i, self.den, self.NOAEC_bird)
        self.sa_mamm_1_s = self.sa_mamm_1(self.first_app_lb, self.a_i, self.den, self.at_mamm, self.fi_mamm, 0.1,
                                          self.ld50_mamm, self.aw_mamm_sm, self.tw_mamm, 0.015)
        self.sa_mamm_2_s = self.sa_mamm_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_mamm,
                                          self.ld50_mamm, self.aw_mamm_sm, self.tw_mamm, 0.015)
        self.sc_mamm_s = self.sc_mamm(self.first_app_lb, self.a_i, self.den, self.NOAEL_mamm, self.aw_mamm_sm,
                                      self.fi_mamm, 0.1, self.tw_mamm, self.ANOAEL_mamm, 0.015)

        self.sa_bird_1_m = self.sa_bird_1(0.1, 0.1, self.aw_bird_md, self.tw_bird_ld50)
        self.sa_bird_2_m = self.sa_bird_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_bird,
                                          self.ld50_bird, self.aw_bird_md, self.tw_bird_ld50, self.x, 0.1)
        self.sc_bird_m = self.sc_bird(self.first_app_lb, self.a_i, self.den, self.NOAEC_bird)
        self.sa_mamm_1_m = self.sa_mamm_1(self.first_app_lb, self.a_i, self.den, self.at_mamm, self.fi_mamm, 0.1,
                                          self.ld50_mamm, self.aw_mamm_md, self.tw_mamm, 0.035)
        self.sa_mamm_2_m = self.sa_mamm_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_mamm,
                                          self.ld50_mamm, self.aw_mamm_md, self.tw_mamm, 0.035)
        self.sc_mamm_m = self.sc_mamm(self.first_app_lb, self.a_i, self.den, self.NOAEL_mamm, self.aw_mamm_md,
                                      self.fi_mamm, 0.1, self.tw_mamm, self.ANOAEL_mamm, 0.035)

        self.sa_bird_1_l = self.sa_bird_1(0.1, 1.0, self.aw_bird_lg, self.tw_bird_ld50)
        self.sa_bird_2_l = self.sa_bird_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_bird,
                                          self.ld50_bird, self.aw_bird_lg, self.tw_bird_ld50, self.x, 1.0)
        self.sc_bird_l = self.sc_bird(self.first_app_lb, self.a_i, self.den, self.NOAEC_bird)
        self.sa_mamm_1_l = self.sa_mamm_1(self.first_app_lb, self.a_i, self.den, self.at_mamm, self.fi_mamm, 0.1,
                                          self.ld50_mamm, self.aw_mamm_lg, self.tw_mamm, 1)
        self.sa_mamm_2_l = self.sa_mamm_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_mamm,
                                          self.ld50_mamm, self.aw_mamm_lg, self.tw_mamm, 1)
        self.sc_mamm_l = self.sc_mamm(self.first_app_lb, self.a_i, self.den, self.NOAEL_mamm, self.aw_mamm_lg,
                                      self.fi_mamm, 0.1, self.tw_mamm, self.ANOAEL_mamm, 1)

        # Table 6
        logging.info("table 6")
        self.EEC_diet_SG = self.EEC_diet(self.C_0_sg, self.C_t_sg, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                         self.day_out)
        self.EEC_diet_TG = self.EEC_diet(self.C_0_tg, self.C_t_tg, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                         self.day_out)
        self.EEC_diet_BP = self.EEC_diet(self.C_0_blp, self.C_t_blp, self.noa, self.first_app_lb, self.a_i, 135,
                                         self.h_l, self.day_out)
        self.EEC_diet_FR = self.EEC_diet(self.C_0_fp, self.C_t_f_p, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                         self.day_out)
        self.EEC_diet_AR = self.EEC_diet(self.C_0_arthro, self.C_t_arhtro, self.noa, self.first_app_lb, self.a_i, 94,
                                         self.h_l, self.day_out)

        # Table 7
        logging.info("table 7")
        self.EEC_dose_bird_SG_sm = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_sm, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_SG_md = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_md, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_SG_lg = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_lg, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_TG_sm = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_sm, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_TG_md = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_md, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_TG_lg = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_lg, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_BP_sm = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_sm, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_BP_md = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_md, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_BP_lg = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_lg, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_FP_sm = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_sm, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_FP_md = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_md, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_FP_lg = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_lg, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_AR_sm = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_sm, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_AR_md = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_md, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_AR_lg = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_lg, self.fi_bird, 0.9, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_SE_sm = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_sm, self.fi_bird, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_SE_md = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_md, self.fi_bird, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_bird_SE_lg = self.EEC_dose_bird(self.EEC_diet, self.aw_bird_lg, self.fi_bird, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)

        # Table 7_add
        self.ARQ_bird_SG_sm = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_sm, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_SG_md = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_md, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_SG_lg = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_lg, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_TG_sm = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_sm, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_TG_md = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_md, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_TG_lg = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_lg, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_BP_sm = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_sm, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_BP_md = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_md, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_BP_lg = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_lg, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_FP_sm = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_sm, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_FP_md = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_md, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_FP_lg = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_lg, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_AR_sm = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_sm, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_AR_md = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_md, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_AR_lg = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_lg, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.8, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_SE_sm = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_sm, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.1, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_SE_md = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_md, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.1, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                 self.day_out)
        self.ARQ_bird_SE_lg = self.ARQ_dose_bird(self.EEC_dose_bird, self.EEC_diet, self.aw_bird_lg, self.fi_bird,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x, 0.1, self.C_0,
                                                 self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                 self.day_out)

        # Table 8
        logging.info("table 8")
        self.ARQ_diet_bird_SG_A = self.ARQ_diet_bird(self.EEC_diet, self.lc50_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 240, self.h_l, self.day_out)
        self.ARQ_diet_bird_SG_C = self.CRQ_diet_bird(self.EEC_diet, self.NOAEC_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 240, self.h_l, self.day_out)
        self.ARQ_diet_bird_TG_A = self.ARQ_diet_bird(self.EEC_diet, self.lc50_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 110, self.h_l, self.day_out)
        self.ARQ_diet_bird_TG_C = self.CRQ_diet_bird(self.EEC_diet, self.NOAEC_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 110, self.h_l, self.day_out)
        self.ARQ_diet_bird_BP_A = self.ARQ_diet_bird(self.EEC_diet, self.lc50_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 135, self.h_l, self.day_out)
        self.ARQ_diet_bird_BP_C = self.CRQ_diet_bird(self.EEC_diet, self.NOAEC_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 135, self.h_l, self.day_out)
        self.ARQ_diet_bird_FP_A = self.ARQ_diet_bird(self.EEC_diet, self.lc50_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 15, self.h_l, self.day_out)
        self.ARQ_diet_bird_FP_C = self.CRQ_diet_bird(self.EEC_diet, self.NOAEC_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 15, self.h_l, self.day_out)
        self.ARQ_diet_bird_AR_A = self.ARQ_diet_bird(self.EEC_diet, self.lc50_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 94, self.h_l, self.day_out)
        self.ARQ_diet_bird_AR_C = self.CRQ_diet_bird(self.EEC_diet, self.NOAEC_bird, self.C_0, self.C_t, self.noa,
                                                     self.first_app_lb, self.a_i, 94, self.h_l, self.day_out)

        # Table 9
        logging.info("table 9")
        self.EEC_dose_mamm_SG_sm = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_sm, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_SG_md = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_md, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_SG_lg = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_lg, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_TG_sm = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_sm, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_TG_md = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_md, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_TG_lg = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_lg, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_BP_sm = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_sm, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_BP_md = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_md, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_BP_lg = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_lg, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_FP_sm = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_sm, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_FP_md = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_md, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_FP_lg = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_lg, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_AR_sm = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_sm, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_AR_md = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_md, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_AR_lg = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_lg, self.fi_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_SE_sm = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_sm, self.fi_mamm, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_SE_md = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_md, self.fi_mamm, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.EEC_dose_mamm_SE_lg = self.EEC_dose_mamm(self.EEC_diet, self.aw_mamm_lg, self.fi_mamm, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)

        # Table 10
        logging.info("table 10")
        self.ARQ_dose_mamm_SG_sm = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_sm,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_SG_sm = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_sm, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 240,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_SG_md = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_md,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_SG_md = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_md, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 240,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_SG_lg = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_lg,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 240, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_SG_lg = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_lg, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 240,
                                                      self.h_l, self.day_out)

        self.ARQ_dose_mamm_TG_sm = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_sm,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_TG_sm = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_sm, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 110,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_TG_md = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_md,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_TG_md = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_md, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 110,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_TG_lg = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_lg,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 110, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_TG_lg = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_lg, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 110,
                                                      self.h_l, self.day_out)

        self.ARQ_dose_mamm_BP_sm = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_sm,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_BP_sm = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_sm, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 135,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_BP_md = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_md,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_BP_md = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_md, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 135,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_BP_lg = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_lg,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 135, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_BP_lg = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_lg, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 135,
                                                      self.h_l, self.day_out)

        self.ARQ_dose_mamm_FP_sm = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_sm,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_FP_sm = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_sm, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 15,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_FP_md = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_md,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_FP_md = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_md, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 15,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_FP_lg = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_lg,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_FP_lg = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_lg, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 15,
                                                      self.h_l, self.day_out)

        self.ARQ_dose_mamm_AR_sm = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_sm,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_AR_sm = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_sm, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 94,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_AR_md = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_md,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_AR_md = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_md, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 94,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_AR_lg = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_lg,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.8, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 94, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_AR_lg = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_lg, self.fi_mamm, self.tw_mamm, 0.8,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 94,
                                                      self.h_l, self.day_out)

        self.ARQ_dose_mamm_SE_sm = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_sm,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_SE_sm = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_sm, self.fi_mamm, self.tw_mamm, 0.1,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 15,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_SE_md = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_md,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_SE_md = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_md, self.fi_mamm, self.tw_mamm, 0.1,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 15,
                                                      self.h_l, self.day_out)
        self.ARQ_dose_mamm_SE_lg = self.ARQ_dose_mamm(self.EEC_dose_mamm, self.EEC_diet, self.at_mamm, self.aw_mamm_lg,
                                                      self.fi_mamm, self.ld50_mamm, self.tw_mamm, 0.1, self.C_0,
                                                      self.C_t, self.noa, self.first_app_lb, self.a_i, 15, self.h_l,
                                                      self.day_out)
        self.CRQ_dose_mamm_SE_lg = self.CRQ_dose_mamm(self.EEC_diet, self.EEC_dose_mamm, self.ANOAEL_mamm,
                                                      self.NOAEL_mamm, self.aw_mamm_lg, self.fi_mamm, self.tw_mamm, 0.1,
                                                      self.C_0, self.C_t, self.noa, self.first_app_lb, self.a_i, 15,
                                                      self.h_l, self.day_out)

        # table 11
        logging.info("table 11")
        if self.lc50_mamm != 'N/A':
            self.ARQ_diet_mamm_SG = self.ARQ_diet_mamm(self.EEC_diet, self.lc50_mamm, self.C_0, self.C_t, self.noa,
                                                       self.first_app_lb, self.a_i, 240, self.h_l, self.day_out)
            self.ARQ_diet_mamm_TG = self.ARQ_diet_mamm(self.EEC_diet, self.lc50_mamm, self.C_0, self.C_t, self.noa,
                                                       self.first_app_lb, self.a_i, 110, self.h_l, self.day_out)
            self.ARQ_diet_mamm_BP = self.ARQ_diet_mamm(self.EEC_diet, self.lc50_mamm, self.C_0, self.C_t, self.noa,
                                                       self.first_app_lb, self.a_i, 135, self.h_l, self.day_out)
            self.ARQ_diet_mamm_FP = self.ARQ_diet_mamm(self.EEC_diet, self.lc50_mamm, self.C_0, self.C_t, self.noa,
                                                       self.first_app_lb, self.a_i, 15, self.h_l, self.day_out)
            self.ARQ_diet_mamm_AR = self.ARQ_diet_mamm(self.EEC_diet, self.lc50_mamm, self.C_0, self.C_t, self.noa,
                                                       self.first_app_lb, self.a_i, 94, self.h_l, self.day_out)
        else:
            self.ARQ_diet_mamm_SG = 'N/A'
            self.ARQ_diet_mamm_TG = 'N/A'
            self.ARQ_diet_mamm_BP = 'N/A'
            self.ARQ_diet_mamm_FP = 'N/A'
            self.ARQ_diet_mamm_AR = 'N/A'

        self.CRQ_diet_mamm_SG = self.CRQ_diet_mamm(self.EEC_diet, self.NOAEC_mamm, self.C_0, self.C_t, self.noa,
                                                   self.first_app_lb, self.a_i, 240, self.h_l, self.day_out)
        self.CRQ_diet_mamm_TG = self.CRQ_diet_mamm(self.EEC_diet, self.NOAEC_mamm, self.C_0, self.C_t, self.noa,
                                                   self.first_app_lb, self.a_i, 110, self.h_l, self.day_out)
        self.CRQ_diet_mamm_BP = self.CRQ_diet_mamm(self.EEC_diet, self.NOAEC_mamm, self.C_0, self.C_t, self.noa,
                                                   self.first_app_lb, self.a_i, 135, self.h_l, self.day_out)
        self.CRQ_diet_mamm_FP = self.CRQ_diet_mamm(self.EEC_diet, self.NOAEC_mamm, self.C_0, self.C_t, self.noa,
                                                   self.first_app_lb, self.a_i, 15, self.h_l, self.day_out)
        self.CRQ_diet_mamm_AR = self.CRQ_diet_mamm(self.EEC_diet, self.NOAEC_mamm, self.C_0, self.C_t, self.noa,
                                                   self.first_app_lb, self.a_i, 94, self.h_l, self.day_out)

        # Table12
        logging.info("table 12")
        self.LD50_rg_bird_sm = self.LD50_rg_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.r_s,
                                                 self.b_w, self.aw_bird_sm, self.at_bird, self.ld50_bird,
                                                 self.tw_bird_ld50, self.x)
        self.LD50_rg_mamm_sm = self.LD50_rg_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.r_s,
                                                 self.b_w, self.aw_mamm_sm, self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_rg_bird_md = self.LD50_rg_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.r_s,
                                                 self.b_w, self.aw_bird_md, self.at_bird, self.ld50_bird,
                                                 self.tw_bird_ld50, self.x)
        self.LD50_rg_mamm_md = self.LD50_rg_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.r_s,
                                                 self.b_w, self.aw_mamm_md, self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_rg_bird_lg = self.LD50_rg_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.r_s,
                                                 self.b_w, self.aw_bird_lg, self.at_bird, self.ld50_bird,
                                                 self.tw_bird_ld50, self.x)
        self.LD50_rg_mamm_lg = self.LD50_rg_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.r_s,
                                                 self.b_w, self.aw_mamm_lg, self.at_mamm, self.ld50_mamm, self.tw_mamm)

        # Table13
        logging.info("table 13")
        self.LD50_rl_bird_sm = self.LD50_rl_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.b_w,
                                                 self.aw_bird_sm, self.at_bird, self.ld50_bird, self.tw_bird_ld50,
                                                 self.x)
        self.LD50_rl_mamm_sm = self.LD50_rl_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.b_w,
                                                 self.aw_mamm_sm, self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_rl_bird_md = self.LD50_rl_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.b_w,
                                                 self.aw_bird_md, self.at_bird, self.ld50_bird, self.tw_bird_ld50,
                                                 self.x)
        self.LD50_rl_mamm_md = self.LD50_rl_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.b_w,
                                                 self.aw_mamm_md, self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_rl_bird_lg = self.LD50_rl_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.b_w,
                                                 self.aw_bird_lg, self.at_bird, self.ld50_bird, self.tw_bird_ld50,
                                                 self.x)
        self.LD50_rl_mamm_lg = self.LD50_rl_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i, self.b_w,
                                                 self.aw_mamm_lg, self.at_mamm, self.ld50_mamm, self.tw_mamm)

        # Table14
        logging.info("table 14")
        self.LD50_bg_bird_sm = self.LD50_bg_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i,
                                                 self.aw_bird_sm, self.at_bird, self.ld50_bird, self.tw_bird_ld50,
                                                 self.x)
        self.LD50_bg_mamm_sm = self.LD50_bg_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i,
                                                 self.aw_mamm_sm, self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_bg_bird_md = self.LD50_bg_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i,
                                                 self.aw_bird_md, self.at_bird, self.ld50_bird, self.tw_bird_ld50,
                                                 self.x)
        self.LD50_bg_mamm_md = self.LD50_bg_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i,
                                                 self.aw_mamm_md, self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_bg_bird_lg = self.LD50_bg_bird(self.Application_type, self.first_app_lb, self.a_i, self.p_i,
                                                 self.aw_bird_lg, self.at_bird, self.ld50_bird, self.tw_bird_ld50,
                                                 self.x)
        self.LD50_bg_mamm_lg = self.LD50_bg_mamm(self.Application_type, self.first_app_lb, self.a_i, self.p_i,
                                                 self.aw_mamm_lg, self.at_mamm, self.ld50_mamm, self.tw_mamm)

        # Table15
        logging.info("table 15")
        self.LD50_bl_bird_sm = self.LD50_bl_bird(self.Application_type, self.first_app_lb, self.a_i, self.aw_bird_sm,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x)
        self.LD50_bl_mamm_sm = self.LD50_bl_mamm(self.Application_type, self.first_app_lb, self.a_i, self.aw_mamm_sm,
                                                 self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_bl_bird_md = self.LD50_bl_bird(self.Application_type, self.first_app_lb, self.a_i, self.aw_bird_md,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x)
        self.LD50_bl_mamm_md = self.LD50_bl_mamm(self.Application_type, self.first_app_lb, self.a_i, self.aw_mamm_md,
                                                 self.at_mamm, self.ld50_mamm, self.tw_mamm)
        self.LD50_bl_bird_lg = self.LD50_bl_bird(self.Application_type, self.first_app_lb, self.a_i, self.aw_bird_lg,
                                                 self.at_bird, self.ld50_bird, self.tw_bird_ld50, self.x)
        self.LD50_bl_mamm_lg = self.LD50_bl_mamm(self.Application_type, self.first_app_lb, self.a_i, self.aw_mamm_lg,
                                                 self.at_mamm, self.ld50_mamm, self.tw_mamm)


    # food intake for birds
    @timefn
    def fi_bird(self, aw_bird, mf_w_bird):
        return (0.648 * (aw_bird ** 0.651)) / (1 - mf_w_bird)

    # food intake for mammals
    @timefn
    def fi_mamm(self, aw_mamm, mf_w_mamm):
        return (0.621 * (aw_mamm ** 0.564)) / (1 - mf_w_mamm)

    # Acute adjusted toxicity value for birds
    @timefn
    def at_bird(self, ld50_bird, aw_bird, tw_bird, x):
        logging.info("at_bird")
        logging.info(ld50_bird)
        logging.info(aw_bird)
        logging.info(tw_bird)
        logging.info(x)
        at_bird_return = ld50_bird * (aw_bird / tw_bird) ** (x - 1)
        return at_bird_return

    # Acute adjusted toxicity value for mammals
    @timefn
    def at_mamm(self, ld50_mamm, aw_mamm, tw_mamm):
        return ld50_mamm * ((tw_mamm / aw_mamm) ** 0.25)

    # Adjusted chronic toxicity (NOAEL) value for mammals
    @timefn
    def ANOAEL_mamm(self, NOAEL_mamm, aw_mamm, tw_mamm):
        return NOAEL_mamm * ((tw_mamm / aw_mamm) ** 0.25)

    # Dietary based EECs
    # Initial concentration from new application
    @timefn
    def conc_initial(self, a_r, a_i, food_multipler):
        conc_new = (a_r * a_i * food_multiplier)
        return conc_new

    # Concentration over time
    @timefn
    def conc_timestep(self, C_ini, h_l):
        return C_ini * np.exp(-(np.log(2) / h_l) * 1)

    # Concentration time series for a selected food item
    @timefn
    def C_timeseries(self, food_multiplier):
        """

        :type self: object
        """
        conc_food = np.zeros((371, 1))  # empty array to hold the concentrations over days
        existing_conc = 0.  # start concentration
        add_conc = 0.  # intermediate concentration calculation
        app_check = False  # checks to see if there are more applications left in the year
        app_counter = 0  # tracks number of applications
        app_day = 0  # app_day tracks day number of the next application
        app_rate = 0.  # application rate of next application
        app_total = 0  # total number of applications
        app_total = len(self.day_out)

        for i in range(0, 371):  # i is day number in the year
            app_check = bool(app_counter <= app_total)
            if app_check:  # check for next application day
                logging.info(self.day_out)
                logging.info(self.rate_out)
                app_day = int(self.day_out[0][app_counter])  # day number of the next application
                logging.info(app_day)
                app_rate = float(self.rate_out[0])  # application rate of next application
                logging.info(app_rate)
                if i == app_day:  # application day
                    if i > 0:  # decay yesterdays concentration
                        existing_conc = self.conc_timestep(conc_temp[i - 1], self.h_l)
                    add_conc = self.conc_initial(app_rate, self.a_i, food_multiplier)  # new application conc
                    conc_food[i] = existing_conc + add_conc  # calculate today's total concentration
                    app_counter += 1  # increment number of applications so far
                elif i > 0:
                    conc_food[i] = self.conc_timestep(C_temp[i - 1],
                                                      self.h_l)  # decay yesterdays concentration if no application
                else:
                    conc_food[i] = 0  # handle first day if no application
        return C_temp

    # concentration over time if application rate or time interval is variable
    # returns max daily concentration, can be multiple applications
    # Dietary based EECs
    @timefn
    def EEC_diet(self, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        # new in trex1.5.1
        logging.info("EEC_diet")
        logging.info("noa")
        logging.info(noa.dtype)
        logging.info(noa)
        logging.info(noa.any())
        logging.info("a_r")
        logging.info(a_r)
        if noa.any() == 1:
            # get initial concentration
            C_temp = C_0(a_r, a_i, para)
            logging.info("C_temp")
            logging.info(C_temp)
            return np.array([C_temp])
        else:
            C_temp = np.ones((371, 1))  # empty array to hold the concentrations over days
            a_p_temp = 0  # application period temp
            noa_temp = 0  # number of existing applications
            dayt = 0
            day_out_l = len(day_out)
            for i in range(0, 371):
                if i == 0:  # first day of application
                    C_temp[i] = C_0(a_r[0], a_i, para)
                    a_p_temp = 0
                    noa_temp += 1
                    dayt += 1
                elif dayt <= day_out_l - 1 and noa_temp <= noa:  # next application day
                    if i == day_out[dayt]:
                        C_temp[i] = C_t(C_temp[i - 1], h_l) + C_0(a_r[dayt], a_i, para)
                        noa_temp += 1
                        dayt += 1
                    else:
                        C_temp[i] = C_t(C_temp[i - 1], h_l)
            logging.info("C_temp")
            logging.info(C_temp)
            max_c_return = max(C_temp)
            logging.info("max_c_return")
            logging.info(max_c_return)
            return max_c_return


    # Dose based EECs for birds
    @timefn
    def EEC_dose_bird(self, EEC_diet, aw_bird, mf_w_bird, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        logging.info("EEC_dose_bird")
        fi_bird_calc = fi_bird(aw_bird, mf_w_bird)
        EEC_diet = EEC_diet(C_0, C_t, noa, a_r, a_i, para, h_l, day_out)
        logging.info(EEC_diet)
        logging.info(fi_bird_calc)
        logging.info(aw_bird)
        EEC_out = EEC_diet * fi_bird_calc / aw_bird
        logging.info("EEC_out")
        logging.info(EEC_out)
        return EEC_out

    # Dose based EECs for granivores birds

    # def EEC_dose_bird_g(EEC_diet, aw_bird, fi_bird, mf_w_bird, C_0, noa, a_r, self.a_i, para, h_l):
    # if para==15:
    # noa = float(noa)
    # #  i_a = float(i_a)
    # aw_bird = float(aw_bird)
    # mf_w_bird = float(mf_w_bird)
    # a_r = float(a_r)
    # a_i = float(a_i)
    # para = float(para)
    # h_l = float(h_l)
    # fi_bird = fi_bird(aw_bird, mf_w_bird)
    # EEC_diet=EEC_diet(C_0, noa, a_r, a_i, para, h_l, day)
    # return (EEC_diet*fi_bird/aw_bird)
    # else:
    # return(0)

    # Dose based EECs for mammals
    @timefn
    def EEC_dose_mamm(self, EEC_diet, aw_mamm, fi_mamm, mf_w_mamm, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        EEC_diet = EEC_diet(C_0, C_t, noa, a_r, a_i, para, h_l, day_out)
        fi_mamm = fi_mamm(aw_mamm, mf_w_mamm)
        return EEC_diet * fi_mamm / aw_mamm

    # Dose based EECs for granivores mammals

    # def EEC_dose_mamm_g(EEC_diet, aw_mamm, fi_mamm, mf_w_mamm, C_0, noa, a_r, a_i, para, h_l):
    # if para==15:
    # aw_mamm = float(aw_mamm)
    # EEC_diet=EEC_diet(C_0, noa, a_r, a_i, para, h_l, day)
    # fi_mamm = fi_mamm(aw_mamm, mf_w_mamm)
    # return (EEC_diet*fi_mamm/aw_mamm)
    # else:
    # return(0)

    # Acute dose-based risk quotients for birds
    @timefn
    def ARQ_dose_bird(self, EEC_dose_bird, EEC_diet, aw_bird, fi_bird, at_bird, ld50_bird, tw_bird, x, mf_w_bird, C_0,
                      C_t, noa, a_r, a_i, para, h_l, day_out):
        EEC_dose_bird = EEC_dose_bird(EEC_diet, aw_bird, fi_bird, mf_w_bird, C_0, C_t, noa, a_r, a_i, para, h_l,
                                      day_out)
        at_bird = at_bird(ld50_bird, aw_bird, tw_bird, x)
        return EEC_dose_bird / at_bird

    # Acute dose-based risk quotients for granivores birds

    # def ARQ_dose_bird_g(EEC_dose_bird, EEC_diet, aw_bird, fi_bird, at_bird, ld50_bird, tw_bird, x, mf_w_bird, C_0, noa, a_r, a_i, para, h_l):
    # if para==15:
    # EEC_dose_bird = EEC_dose_bird(EEC_diet, aw_bird, fi_bird, mf_w_bird, C_0, noa, a_r, a_i, para, h_l)
    # at_bird = at_bird(ld50_bird,aw_bird,tw_bird,x)
    # return (EEC_dose_bird/at_bird)
    # else:
    # return (0)

    # Acute dose-based risk quotients for mammals
    @timefn
    def ARQ_dose_mamm(self, EEC_dose_mamm, EEC_diet, at_mamm, aw_mamm, fi_mamm, ld50_mamm, tw_mamm, mf_w_mamm, C_0, C_t,
                      noa, a_r, a_i, para, h_l, day_out):
        EEC_dose_mamm = EEC_dose_mamm(EEC_diet, aw_mamm, fi_mamm, mf_w_mamm, C_0, C_t, noa, a_r, a_i, para, h_l,
                                      day_out)
        at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
        return EEC_dose_mamm / at_mamm

    # Acute dose-based risk quotients for granivores mammals
    # def ARQ_dose_mamm_g(EEC_dose_mamm, at_mamm, aw_mamm, ld50_mamm, tw_mamm, mf_w_mamm, C_0, noa, a_r, a_i, para, h_l):
    # if para==15:
    # EEC_dose_mamm = EEC_dose_mamm(EEC_diet, aw_mamm, fi_mamm, mf_w_mamm, C_0, noa, a_r, a_i, para, h_l)
    # at_mamm = at_mamm(ld50_mamm,aw_mamm,tw_mamm)
    # return (EEC_dose_mamm/at_mamm)
    # else:
    # return(0)

    # Acute dietary-based risk quotients for birds
    @timefn
    def ARQ_diet_bird(self, EEC_diet, lc50_bird, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        EEC_diet = EEC_diet(C_0, C_t, noa, a_r, a_i, para, h_l, day_out)
        return EEC_diet / lc50_bird

    # Acute dietary-based risk quotients for mammals
    @timefn
    def ARQ_diet_mamm(self, EEC_diet, lc50_mamm, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        EEC_diet = EEC_diet(C_0, C_t, noa, a_r, a_i, para, h_l, day_out)
        return EEC_diet / lc50_mamm

    # Chronic dietary-based risk quotients for birds
    @timefn
    def CRQ_diet_bird(self, EEC_diet, NOAEC_bird, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        EEC_diet = EEC_diet(C_0, C_t, noa, a_r, a_i, para, h_l, day_out)
        return EEC_diet / NOAEC_bird

    # Chronic dietary-based risk quotients for mammals
    @timefn
    def CRQ_diet_mamm(self, EEC_diet, NOAEC_mamm, C_0, C_t, noa, a_r, a_i, para, h_l, day_out):
        EEC_diet = EEC_diet(C_0, C_t, noa, a_r, a_i, para, h_l, day_out)
        return EEC_diet / NOAEC_mamm

    # Chronic dose-based risk quotients for mammals
    @timefn
    def CRQ_dose_mamm(self, EEC_diet, EEC_dose_mamm, ANOAEL_mamm, NOAEL_mamm, aw_mamm, fi_mamm, tw_mamm, mf_w_mamm, C_0,
                      C_t, noa, a_r, a_i, para, h_l, day_out):
        ANOAEL_mamm = ANOAEL_mamm(NOAEL_mamm, aw_mamm, tw_mamm)
        EEC_dose_mamm = EEC_dose_mamm(EEC_diet, aw_mamm, fi_mamm, mf_w_mamm, C_0, C_t, noa, a_r, a_i, para, h_l,
                                      day_out)
        return EEC_dose_mamm / ANOAEL_mamm

    # Chronic dose-based risk quotients for granviores mammals
    # def CRQ_dose_mamm_g(EEC_diet, EEC_dose_mamm, ANOAEL_mamm, NOAEL_mamm, aw_mamm, tw_mamm, mf_w_mamm, noa, a_r, a_i, para, h_l):
    #     if para==15:    
    #         ANOAEL_mamm=ANOAEL_mamm(NOAEL_mamm,aw_mamm,tw_mamm)
    #         EEC_dose_mamm = EEC_dose_mamm(EEC_diet, aw_mamm, fi_mamm, mf_w_mamm, C_0, noa, a_r, a_i, para, h_l)     
    #         return (EEC_dose_mamm/ANOAEL_mamm)
    #     else:
    #         return (0)

    # LD50ft-2 for row/band/in-furrow granular birds
    @timefn
    def LD50_rg_bird(self, Application_type, a_r, a_i, p_i, r_s, b_w, aw_bird, at_bird, ld50_bird, tw_bird, x):
        if Application_type == 'Row/Band/In-furrow-Granular':
            at_bird = at_bird(ld50_bird, aw_bird, tw_bird, x)
            # print 'r_s', r_s
            n_r = (43560 ** 0.5) / r_s
            # print 'n_r=', n_r
            # print 'a_r=', a_r
            # print 'b_w=', b_w
            # print 'p_i=', p_i
            # print 'a_i', a_i
            # print 'class a_r', type(a_r)
            expo_rg_bird = (max(a_r) * a_i * 453590.0) / (n_r * (43560.0 ** 0.5) * b_w) * (1 - p_i)
            return expo_rg_bird / (at_bird * (aw_bird / 1000.0))
        else:
            return 0

    # LD50ft-2 for row/band/in-furrow liquid birds
    @timefn
    def LD50_rl_bird(self, Application_type, a_r, a_i, p_i, b_w, aw_bird, at_bird, ld50_bird, tw_bird, x):
        if Application_type == 'Row/Band/In-furrow-Liquid':
            at_bird = at_bird(ld50_bird, aw_bird, tw_bird, x)
            expo_rl_bird = ((max(a_r) * 28349 * a_i) / (1000 * b_w)) * (1 - p_i)
            return expo_rl_bird / (at_bird * (aw_bird / 1000.0))
        else:
            return 0

    # LD50ft-2 for row/band/in-furrow granular mammals
    @timefn
    def LD50_rg_mamm(self, Application_type, a_r, a_i, p_i, r_s, b_w, aw_mamm, at_mamm, ld50_mamm, tw_mamm):
        if Application_type == 'Row/Band/In-furrow-Granular':
            # a_r = max(first_app_lb)
            at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
            n_r = (43560 ** 0.5) / r_s
            expo_rg_mamm = (max(a_r) * a_i * 453590) / (n_r * (43560 ** 0.5) * b_w) * (1 - p_i)
            return expo_rg_mamm / (at_mamm * (aw_mamm / 1000.0))
        else:
            return 0

    # LD50ft-2 for row/band/in-furrow liquid mammals
    @timefn
    def LD50_rl_mamm(self, Application_type, a_r, a_i, p_i, b_w, aw_mamm, at_mamm, ld50_mamm, tw_mamm):
        if Application_type == 'Row/Band/In-furrow-Liquid':
            at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
            expo_rl_bird = ((max(a_r) * 28349 * a_i) / (1000 * b_w)) * (1 - p_i)
            return expo_rl_bird / (at_mamm * (aw_mamm / 1000.0))
        else:
            return 0

    # LD50ft-2 for broadcast granular birds
    @timefn
    def LD50_bg_bird(self, Application_type, a_r, a_i, p_i, aw_bird, at_bird, ld50_bird, tw_bird, x):
        if Application_type == 'Broadcast-Granular':
            at_bird = at_bird(ld50_bird, aw_bird, tw_bird, x)
            expo_bg_bird = ((max(a_r) * a_i * 453590) / 43560)
            return expo_bg_bird / (at_bird * (aw_bird / 1000.0))
        else:
            return 0

    # LD50ft-2 for broadcast liquid birds
    @timefn
    def LD50_bl_bird(self, Application_type, a_r, a_i, aw_bird, at_bird, ld50_bird, tw_bird, x):
        if Application_type == 'Broadcast-Liquid':
            at_bird = at_bird(ld50_bird, aw_bird, tw_bird, x)
            # expo_bl_bird=((max(a_r)*28349*a_i)/43560)*(1-p_i)
            expo_bl_bird = ((max(a_r) * 453590 * a_i) / 43560)
            return expo_bl_bird / (at_bird * (aw_bird / 1000.0))
        else:
            return 0

    # LD50ft-2 for broadcast granular mammals
    @timefn
    def LD50_bg_mamm(self, Application_type, a_r, a_i, p_i, aw_mamm, at_mamm, ld50_mamm, tw_mamm):
        if Application_type == 'Broadcast-Granular':
            at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
            expo_bg_mamm = ((max(a_r) * a_i * 453590) / 43560)
            return expo_bg_mamm / (at_mamm * (aw_mamm / 1000.0))
        else:
            return 0

    # LD50ft-2 for broadcast liquid mammals
    @timefn
    def LD50_bl_mamm(self, Application_type, a_r, a_i, aw_mamm, at_mamm, ld50_mamm, tw_mamm):
        if Application_type == 'Broadcast-Liquid':
            at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
            # expo_bl_mamm=((max(a_r)*28349*a_i)/43560)*(1-p_i)
            expo_bl_mamm = ((max(a_r) * a_i * 453590) / 43560)
            return expo_bl_mamm / (at_mamm * (aw_mamm / 1000.0))
        else:
            return 0

    # Seed treatment acute RQ for birds method 1
    #@timefn
    def sa_bird_1(self, mf_w_bird, nagy_bird_coef, aw_bird, tw_bird):
        #logging
        logging.info("sa_bird_1")
        logging.info(self.ld50_bird)
        logging.info(aw_bird)
        logging.info(tw_bird)
        logging.info(self.x)

        #setup panda series
        at_bird_temp = pd.Series(name="at_bird_temp")
        fi_bird_temp = pd.Series(name="fi_bird_temp")
        m_s_a_r_temp = pd.Series(name="m_s_a_r_temp")
        nagy_bird_temp = pd.Series(name="nagy_bird_temp")
        sa_bird_1_return = pd.Series(name="sa_bird_1_return")

        #run calculations
        at_bird_temp = self.at_bird(self.ld50_bird, aw_bird, tw_bird, self.x)
        fi_bird_temp = self.fi_bird(aw_bird, mf_w_bird)
        # maximum seed application rate=application rate*10000
        m_s_a_r_temp = ((self.first_app_lb * self.a_i) / 128.) * self.den * 10000
        nagy_bird_temp = fi_bird_temp * 0.001 * m_s_a_r_temp / nagy_bird_coef
        sa_bird_1_return = nagy_bird_temp / at_bird_temp
        return sa_bird_1_return

        # Seed treatment acute RQ for birds method 2

    #self.sa_bird_2_s = self.sa_bird_2(self.first_app_lb, self.a_i, self.den, self.m_s_r_p, self.at_bird, self.ld50_bird, self.aw_bird_sm, self.tw_bird_ld50, self.x, 0.02)
    @timefn
    def sa_bird_2(self, a_r_p, a_i, den, m_s_r_p, at_bird, ld50_bird, aw_bird, tw_bird, x, nagy_bird_coef):
        at_bird = at_bird(ld50_bird, aw_bird, tw_bird, x)
        m_a_r = (m_s_r_p * ((a_i * a_r_p) / 128) * den) / 100  #maximum application rate
        av_ai = m_a_r * 1e6 / (43560 * 2.2)
        return av_ai / (at_bird * nagy_bird_coef)

        # Seed treatment chronic RQ for birds

    @timefn
    def sc_bird(self, a_r_p, a_i, den, NOAEC_bird):
        m_s_a_r = ((a_r_p * a_i) / 128) * den * 10000  #maximum seed application rate=application rate*10000
        return m_s_a_r / NOAEC_bird

        # Seed treatment acute RQ for mammals method 1

    @timefn
    def sa_mamm_1(self, a_r_p, a_i, den, at_mamm, fi_mamm, mf_w_bird, ld50_mamm, aw_mamm, tw_mamm, nagy_mamm_coef):
        at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
        fi_mamm = fi_mamm(aw_mamm, mf_w_bird)
        m_s_a_r = ((a_r_p * a_i) / 128) * den * 10000  #maximum seed application rate=application rate*10000
        nagy_mamm = fi_mamm * 0.001 * m_s_a_r / nagy_mamm_coef
        return nagy_mamm / at_mamm

        # Seed treatment acute RQ for mammals method 2

    @timefn
    def sa_mamm_2(self, a_r_p, a_i, den, m_s_r_p, at_mamm, ld50_mamm, aw_mamm, tw_mamm, nagy_mamm_coef):
        at_mamm = at_mamm(ld50_mamm, aw_mamm, tw_mamm)
        m_a_r = (m_s_r_p * ((a_r_p * a_i) / 128) * den) / 100  #maximum application rate
        av_ai = m_a_r * 1000000 / (43560 * 2.2)
        return av_ai / (at_mamm * nagy_mamm_coef)

        # Seed treatment chronic RQ for mammals

    @timefn
    def sc_mamm(self, a_r_p, a_i, den, NOAEL_mamm, aw_mamm, fi_mamm, mf_w_bird, tw_mamm, ANOAEL_mamm, nagy_mamm_coef):
        ANOAEL_mamm = ANOAEL_mamm(NOAEL_mamm, aw_mamm, tw_mamm)
        fi_mamm = fi_mamm(aw_mamm, mf_w_bird)
        m_s_a_r = ((a_r_p * a_i) / 128) * den * 10000  #maximum seed application rate=application rate*10000
        nagy_mamm = fi_mamm * 0.001 * m_s_a_r / nagy_mamm_coef
        return nagy_mamm / ANOAEL_mamm
