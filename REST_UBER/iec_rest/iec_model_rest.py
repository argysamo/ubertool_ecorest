# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import logging
from functools import wraps
import time

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        logging.info(t1)
        logging.info(t2)
        print("sip_model_rest.py@timefn: " + fn.func_name + " took " +
            "{:.6f}".format(t2-t1) + " seconds")
        return result
    return measure_time


class iec(object):
    @timefn
    def __init__(self, run_type, pd_obj, pd_obj_exp):
        # run_type can be single, batch or qaqc
        self.run_type = run_type

        #Inputs: Assign object attribute variables from the input Pandas DataFrame
        self.dose_response = pd_obj['dose_response']
        self.LC50 = pd_obj['LC50']
        self.threshold = pd_obj['threshold']

        # Outputs: Assign object attribute variables to Pandas Series
        self.z_score_f_out = pd.Series(name = "z_score_f_out")
        self.F8_f_out = pd.Series(name = "F8_f_out")
        self.chance_f_out = pd.Series(name = "chance_f_out")

        # Execute model methods
        self.run_methods()

        # Create DataFrame containing output value Series
        pd_obj_out = pd.DataFrame({
            'z_score_f_out' : self.z_score_f_out,
            'F8_f_out' : self.F8_f_out,
            'chance_f_out' : self.chance_f_out
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

    def run_methods(self):
        try:
            self.z_score_f()
            self.F8_f()
            self.chance_f()


    # Begin model methods
    @timefn
    def z_score_f(self):
        # if self.dose_response < 0:
        #     raise ValueError\
        #     ('self.dose_response=%g is a non-physical value.' % self.dose_response)
        # if self.LC50 < 0:
        #     raise ValueError\
        #     ('self.LC50=%g is a non-physical value.' % self.LC50)
        # if self.threshold < 0:
        #     raise ValueError
        #     ('self.threshold=%g is a non-physical value.' % self.threshold)
        # if self.z_score_f_out == -1:
        self.z_score_f_out = self.dose_response * (math.log10(self.LC50 * self.threshold) - math.log10(self.LC50))
        return self.z_score_f_out
        
    def F8_f(self):
        # if self.z_score_f_out == None:
        #     raise ValueError\
        #     ('z_score_f variable equals None and therefor this function cannot be run.')
        # if self.F8_f_out == -1:
        self.F8_f_out = 0.5 * math.erfc(-self.z_score_f_out/math.sqrt(2))
        if self.F8_f_out == 0:
            self.F8_f_out = 10^-16
        else:
            self.F8_f_out = self.F8_f_out
        return self.F8_f_out
        
    def chance_f(self):
        # if self.F8_f_out == None:
        #     raise ValueError\
        #     ('F8_f variable equals None and therefor this function cannot be run.')
        # if self.chance_f_out == -1:
        self.chance_f_out = 1 / self.F8_f_out
        return self.chance_f_out

