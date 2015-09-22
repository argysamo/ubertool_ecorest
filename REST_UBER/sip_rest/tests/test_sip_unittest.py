import unittest
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
#following works when running test script in parent directory as package:
# python -m tests.test_sip_unittest
#import sip_model_rest as sip_model
#following works for running as nosetests from parent directory:
from .. import sip_model_rest as sip_model

# create empty pandas dataframes to create empty sip object for testing
df_empty = pd.DataFrame()
# create an empty sip object
sip_empty = sip_model.sip("empty", df_empty, df_empty)

test = {}

class TestSip(unittest.TestCase):
    def setup(self):
        pass
        # sip2 = sip_model.sip(0, pd_obj_inputs, pd_obj_exp_out)
        # setup the test as needed
        # e.g. pandas to open sip qaqc csv
        #  Read qaqc csv and create pandas DataFrames for inputs and expected outputs

    def teardown(self):
        pass
        # teardown called after each test
        # e.g. maybe write test results to some text file

    def test_fw_bird(self):
        result = sip_empty.fw_bird()
        npt.assert_array_almost_equal(result, 0.0162, 4, '', True)
        return

    def test_fw_mamm(self):
        result = sip_empty.fw_mamm()
        npt.assert_array_almost_equal(result, 0.172, 4, '', True)
        return

    def test_dose_bird(self):
        #(self.fw_bird_out * self.solubility)/(self.bodyweight_assessed_bird / 1000.)
        sip_empty.fw_bird_out = pd.Series([10.], dtype='int')
        sip_empty.solubility = pd.Series([100.], dtype='int')
        sip_empty.bodyweight_assessed_bird = pd.Series([1.], dtype='int')
        result = sip_empty.dose_bird()
        npt.assert_array_almost_equal(result, 1000000., 4, '', True)
        return

    def test_dose_mamm(self):
        #(self.fw_mamm_out * self.solubility)/(self.bodyweight_assessed_mammal / 1000)
        sip_empty.fw_mamm_out = pd.Series([20.], dtype='int')
        sip_empty.solubility = pd.Series([400.], dtype='int')
        sip_empty.bodyweight_assessed_mammal = pd.Series([1.], dtype='int')
        result = sip_empty.dose_mamm()
        npt.assert_array_almost_equal(result, 8000000., 4, '', True)
        return

    def test_at_bird(self):
        #(self.ld50_avian_water) * ((self.bodyweight_assessed_bird / self.bodyweight_tested_bird)**(self.mineau_scaling_factor - 1.))
        sip_empty.ld50_avian_water = pd.Series([2000.], dtype='int')
        sip_empty.bodyweight_assessed_bird = pd.Series([100.], dtype='int')
        sip_empty.bodyweight_tested_bird = pd.Series([200.], dtype='int')
        sip_empty.mineau_scaling_factor = pd.Series([2.], dtype='int')
        result = sip_empty.at_bird()
        npt.assert_array_almost_equal(result, 1000., 4, '', True)
        return

    def test_at_mamm(self):
        #(self.ld50_mammal_water) * ((self.bodyweight_tested_mammal / self.bodyweight_assessed_mammal)**0.25)
        sip_empty.ld50_mammal_water = pd.Series([10.], dtype='int')
        sip_empty.bodyweight_tested_mammal = pd.Series([100.], dtype='int')
        sip_empty.bodyweight_assessed_mammal = pd.Series([200.], dtype='int')
        result = sip_empty.at_mamm()
        npt.assert_array_almost_equal(result, 8.408964, 4, '', True)
        return

    def test_fi_bird(self):
        #0.0582 * ((bw_grams / 1000.)**0.651)
        sip_empty.bw_grams = pd.Series([100.], dtype='int')
        result = sip_empty.fi_bird(sip_empty.bw_grams)
        npt.assert_array_almost_equal(result, 0.012999, 4, '', True)
        return

    def test_act(self):
        #(self.noael_mammal_water) * ((self.bodyweight_tested_mammal / self.bodyweight_assessed_mammal)**0.25)
        sip_empty.noael_mammal_water = pd.Series([10.], dtype='int')
        sip_empty.bodyweight_tested_mammal = pd.Series([500.], dtype='int')
        sip_empty.bodyweight_assessed_mammal = pd.Series([400.], dtype='int')
        result = sip_empty.act()
        npt.assert_array_almost_equal(result, 10.5737, 4, '', True)
        return

#Weird equation. Let's talk about this one.
    def test_det(self):
        """
        Dose Equiv. Toxicity:

        The FI value (kg-diet) is multiplied by the reported NOAEC (mg/kg-diet) and then divided by
        the test animal's body weight to derive the dose-equivalent chronic toxicity value (mg/kg-bw):

        Dose Equiv. Toxicity = (NOAEC * FI) / BW

        NOTE: The user enters the lowest available NOAEC for the mallard duck, for the bobwhite quail,
        and for any other test species. The model calculates the dose equivalent toxicity values for
        all of the modeled values (Cells F20-24 and results worksheet) and then selects the lowest dose
        equivalent toxicity value to represent the chronic toxicity of the chemical to birds.
        """
        # result =
        # self.assertEquals(result, )
        # return

#Amber working
    # def test_det_duck(self):
    # # det_duck = (self.noaec_duck * self.fi_bird(1580.)) / (1580. / 1000.)
    #     sip_empty.noaec_duck = pd.Series([1.], dtype='int')
    #     sip_empty.fi_bird = pd.Series([1.], dtype='int')
    #     result = sip_empty.det_duck()
    #     npt.assert_array_almost_equal(result, 1000., 4, '', True)

#Amber working
    # def test_det_quail(self):
    # # det_quail = (self.noaec_quail * self.fi_bird(178.)) / (178. / 1000.)
    #     sip_empty.noaec_quail = pd.Series([1.], dtype='int')
    #     sip_empty.fi_bird = pd.Series([1.], dtype='int')
    #     result = sip_empty.det_quail()
    #     npt.assert_array_almost_equal(result, 1000., 4, '', True)

#Carmen
#        det_other_1 = (self.noaec_bird_other_1 * self.fi_bird(self.bodyweight_bird_other_1)) / (self.bodyweight_bird_other_1 / 1000.)
#        det_other_2 = (self.noaec_bird_other_2 * self.fi_bird(self.bodyweight_bird_other_1)) / (self.bodyweight_bird_other_1 / 1000.)

#Carmen
    # def test_det_other_1(self):
    #     sip_empty.noaec_bird_other_1 = pd.Series([400.]) # mg/kg-diet
    #     sip_empty.bodyweight_bird_other_1 = pd.Series([100]) # grams
    #     result = sip_empty.det_other_1()
    #     npt.assert_array_almost_equal(result, 4666, 4)
    #     return

    def test_acute_bird(self):
        # self.acute_bird_out = self.dose_bird_out / self.at_bird_out
        sip_empty.dose_bird_out = pd.Series([100.], dtype='int')
        sip_empty.at_bird_out = pd.Series([10.], dtype='int')
        result = sip_empty.acute_bird()
        npt.assert_array_almost_equal(result, 10., 4)
        return

    def test_acuconb(self):
        """
        Message stating whether or not a risk is present
        """
        # if self.acuconb_out == -1:
        #     if self.acute_bird_out == None:
        #         raise ValueError\
        #         ('acute_bird variable equals None and therefor this function cannot be run.')
            # if self.acute_bird_out < 0.1:
            #     self.acuconb_out = ('Drinking water exposure alone is NOT a potential concern for birds')
            # else:
            #     self.acuconb_out = ('Exposure through drinking water alone is a potential concern for birds')
        sip_empty.acute_bird_out = pd.Series([0.2])
        result = sip_empty.acuconb()
        exp = pd.Series(["Exposure through drinking water alone is a potential concern for birds"])
        pdt.assert_series_equal(result, exp)
        return

    def test_acute_mamm(self):
        # self.acute_mamm_out = self.dose_mamm_out / self.at_mamm_out
        sip_empty.dose_mamm_out = pd.Series([100.], dtype='int')
        sip_empty.at_mamm_out = pd.Series([10.], dtype='int')
        result = sip_empty.acute_mamm()
        npt.assert_array_almost_equal(result, 10., 4)
        return

    def test_acuconm(self):
        """
        Message stating whether or not a risk is present
        """
        # if self.acuconm_out == -1:
        #     if self.acute_mamm_out == None:
        #         raise ValueError\
        #         ('acute_mamm variable equals None and therefor this function cannot be run.')
        #     if self.acute_mamm_out < 0.1:
        #         self.acuconm_out = ('Drinking water exposure alone is NOT a potential concern for mammals')
        #     else:
        #         self.acuconm_out = ('Exposure through drinking water alone is a potential concern for mammals')
        #     return self.acuconm_out
        sip_empty.acute_mamm_out = pd.Series([0.2])
        result = sip_empty.acuconm()
        exp = pd.Series(["Exposure through drinking water alone is a potential concern for mammals"])
        pdt.assert_series_equal(result, exp)
        return

    def test_chron_bird(self):
    #self.chron_bird_out = self.dose_bird_out / self.det_out
        sip_empty.dose_bird_out = pd.Series([5.], dtype='int')
        sip_empty.det_out = pd.Series([10.], dtype='int')
        result = sip_empty.chron_bird()
        npt.assert_array_almost_equal(result, 0.5, 4, '', True)
        return

    def test_chronconb(self):
    #boolean = self.chron_bird_out < 1
        #self.chronconb_out = boolean.map(lambda x:
         #   'Drinking water exposure alone is NOT a potential concern for birds' if x == True
         #   else 'Exposure through drinking water alone is a potential concern for birds')
        sip_empty.chron_bird_out = pd.Series([3])
        result = sip_empty.chronconb()
        exp = pd.Series(["Exposure through drinking water alone is a potential concern for birds"])
        pdt.assert_series_equal(result, exp)
        return

    def test_chron_mamm(self):
         # self.chron_mamm_out = self.dose_mamm_out / self.act_out
        sip_empty.dose_mamm_out = pd.Series([8.], dtype='int')
        sip_empty.act_out = pd.Series([4.], dtype='int')
        result = sip_empty.chron_mamm()
        npt.assert_array_almost_equal(result, 2, 4, '', True)
        return

    def test_chronconm(self):
        # self.chronconm_out = boolean.map(lambda x:
         #   'Drinking water exposure alone is NOT a potential concern for mammals' if x == True
         #   else 'Exposure through drinking water alone is a potential concern for mammals')
        sip_empty.chron_mamm_out =  pd.Series([0.5])
        result = sip_empty.chronconm()
        exp = pd.Series(["Drinking water exposure alone is NOT a potential concern for mammals"])
        pdt.assert_series_equal(result, exp)
        return

#Note: commented-out rows contain output files that are not running properly in the subsequent blackbox_method test.
    def test_blackbox_method(self):
         self.blackbox_method_int('dose_bird')
         self.blackbox_method_int('dose_mamm')
         self.blackbox_method_int('at_bird')
         self.blackbox_method_int('at_mamm')
         self.blackbox_method_int('fi_bird')
        # self.blackbox_method_int('det')
         self.blackbox_method_int('act')
         self.blackbox_method_int('acute_bird')
         self.blackbox_method_str('acuconb')
         self.blackbox_method_int('acute_mamm')
         self.blackbox_method_str('acuconm')
         self.blackbox_method_int('chron_bird')
         self.blackbox_method_str('chronconb')
         self.blackbox_method_int('chron_mamm')
         self.blackbox_method_str('chronconm')

    def blackbox_method_int(self, output):
        """
        Helper method to reuse code for testing numpy array outputs from SIP model
        :param output: String; Pandas Series name (e.g. column name) without '_out'
        :return:
        """
        result = sip_calc.pd_obj_out[output + "_out"]
        expected = sip_calc.pd_obj_exp[output + "_exp"]
        npt.assert_array_almost_equal(result, expected, 4, '', True)

    def blackbox_method_str(self, output):
        result = sip_calc.pd_obj_out[output + "_out"]
        expected = sip_calc.pd_obj_exp[output + "_exp"]
        npt.assert_array_equal(result, expected)

# unittest will
# 1) call the setup method,
# 2) then call every method starting with "test",
# 3) then the teardown method
if __name__ == '__main__':
    unittest.main()
    #pass