# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging


class earthworm(object):
    def __init__(self, run_type, pd_obj, pd_obj_exp):
        # run_type can be single, batch or qaqc
        self.run_type = run_type

        # Inputs: Assign object attribute variables from the input Pandas DataFrame
        self.k_ow = pd_obj['k_ow']
        self.l_f_e = pd_obj['l_f_e']
        self.c_s = pd_obj['c_s']
        self.k_d = pd_obj['k_d']
        self.p_s = pd_obj['p_s']
        self.c_w = pd_obj['c_w']
        self.m_w = pd_obj['m_w']
        self.p_e = pd_obj['p_e']

        # Create DataFrame containing output value Series
        pd_obj_out = pd.DataFrame({
            self.earthworm_fugacity_out = -1
        })

        self.run_methods()

        # Callable from Bottle that returns JSON
        self.json = self.json(pd_obj, pd_obj_out, pd_obj_exp)

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
        self.earthworm_fugacity()

    def earthworm_fugacity(self):
        if self.earthworm_fugacity_out == -1:
            try:
                self.k_ow = float(self.k_ow)
                self.l_f_e = float(self.l_f_e)
                self.c_s = float(self.c_s)
                self.k_d = float(self.k_d)
                self.p_s = float(self.p_s)
                self.c_w = float(self.c_w)
                self.m_w = float(self.m_w)
                self.p_e = float(self.p_e)
            except ValueError:
                raise ValueError\
                ('The octanol to water partition coefficient must be a real number, not "%g"' % self.k_ow)
            except ValueError:
                raise ValueError\
                ('The lipid fraction of earthworm must be a real number, not "%g"' % self.l_f_e)
            except ValueError:
                raise ValueError\
                ('The chemical concentration in soil must be a real number, not "%g"' % self.c_s)
            except ValueError:
                raise ValueError\
                ('The soil partitioning coefficient must be a real number, not "%g"' % self.k_d)
            except ValueError:
                raise ValueError\
                ('The bulk density of soil must be a real number, not "%g"' % self.p_s)
            except ValueError:
                raise ValueError\
                ('The chemical concentration in pore water of soil must be a real number, not "%g"' % self.c_w)
            except ValueError:
                raise ValueError\
                ('The molecular weight of chemical must be a real number, not "%g"' % self.m_w)
            except ValueError:
                raise ValueError\
                ('The density of earthworm must be a real number, not "%g"' % self.p_e)
            if self.k_ow < 0:
                raise ValueError\
                ('self.k_ow=%g is a non-physical value.' % self.k_ow)
            if self.l_f_e < 0:
                raise ValueError\
                ('self.l_f_e=%g is a non-physical value.' % self.l_f_e)
            if self.l_f_e > 1:
                raise ValueError\
                ('self.l_f_e=%g is a non-physical value.' % self.l_f_e)
            if self.c_s < 0:
                raise ValueError\
                ('self.c_s=%g is a non-physical value.' % self.c_s)
            if self.k_d < 0:
                raise ValueError\
                ('self.k_d=%g is a non-physical value.' % self.k_d)
            if self.p_s < 0:
                raise ValueError\
                ('self.p_s=%g is a non-physical value.' % self.p_s)
            if self.c_w < 0:
                raise ValueError\
                ('self.c_w=%g is a non-physical value.' % self.c_w)
            if self.m_w < 0:
                raise ValueError\
                ('self.m_w=%g is a non-physical value.' % self.m_w)
            if self.p_e < 0:
                raise ValueError\
                ('self.p_e=%g is a non-physical value.' % self.p_e)
            self.earthworm_fugacity_out = self.k_ow*self.l_f_e*(self.c_s/(self.k_d*self.p_s)+self.c_w)*self.m_w/self.p_e
        return self.earthworm_fugacity_out
