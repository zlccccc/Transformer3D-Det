from ..subfile_init import sub_model_entry

module_dict = {
    'votenet': 'votenet.votenet',
    'scanrefer': 'voterefer.refernet'
}


def model_entry(config):
    return sub_model_entry(config, __file__, module_dict)
