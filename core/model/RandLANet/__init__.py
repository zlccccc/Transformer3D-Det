from ..subfile_init import sub_model_entry

module_dict = {
    'RandLANetv1': 'RandLANetv1.RandLANetv1',
    'RandLANetv2': 'RandLANetv2.RandLANetv2',
    'SPConvv1': 'SPConvv1.SPConvv1'
}


def model_entry(config):
    return sub_model_entry(config, __file__, module_dict)
