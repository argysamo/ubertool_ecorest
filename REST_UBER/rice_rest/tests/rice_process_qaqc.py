# needs to be run whenever the qaqc csv is updated
csv_path = "./rice_qaqc.csv"
csv_path2 = "./rice_qaqc2.csv"
csv_in = "./rice_qaqc_in_transpose.csv"
csv_exp = "./rice_qaqc_exp_transpose.csv"
import pandas as pd

try:
    pd_obj_inputs = pd.read_csv(csv_path, index_col=0, header=None, skiprows=1, skipfooter=8, engine='python')
    pd_obj_inputs = pd_obj_inputs.drop(labels=pd_obj_inputs.columns[range(4)], axis=1)
    pd_obj_inputs.index.name = None
    pd_obj_inputs.columns -= 5
    pd_obj_inputs_transposed = pd_obj_inputs.transpose()
    pd_obj_inputs_transposed.to_csv(csv_in)

    pd_obj_exp_out = pd.read_csv(csv_path, index_col=0, header=None, skiprows=13, engine='python')
    pd_obj_exp_out = pd_obj_exp_out.drop(labels=pd_obj_exp_out.columns[range(4)], axis=1)
    pd_obj_exp_out.index.name = None
    pd_obj_exp_out.columns -= 5
    pd_obj_exp_out_transposed = pd_obj_exp_out.transpose()
    pd_obj_exp_out_transposed.to_csv(csv_exp)
except Exception as e:
    print e.message