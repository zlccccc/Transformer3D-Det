from ..subfile_init import sub_model_entry

# must under core.model package
module_dict = {
    'pointnet': 'Pointnet.PointnetInit',
    'pointnet++': 'PointnetPlus.PointnetPlus',
    'pointnet++2': 'PointnetPlusSSG.PointnetPlusSSG',
    'pointnet_partseg': 'PointnetPlusPartSeg.PointnetPlusPartSeg',
    'pointnet_partsegv2': 'PointnetPlusPartSegv2.PointnetPlusPartSegv2',
    'pointnet_partsegHR': 'PointHRNetPartSeg.PointnetPlusPartSegHR',
    'pointnet_partsegHRv2': 'PointHRNetPartSegv2.PointnetPlusPartSegHRv2',
}


def model_entry(config):
    return sub_model_entry(config, __file__, module_dict)
