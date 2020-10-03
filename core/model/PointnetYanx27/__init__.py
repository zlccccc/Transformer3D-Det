from ..subfile_init import sub_model_entry

module_dict = {
    'pointnetInitial': 'PointnetPlus.PointnetPlusInitial',
    'pointnetSSGInitial': 'PointnetSSG.PointnetPlusSSGInitial',
    'pointnetSSGInitialSaveFeature': 'PointnetSSGSaveFeature.PointnetPlusSSGInitialSaveFeature',
    'pointnetPlane': 'PointnetPlane.PointnetPlusPlane',
    'pointnetPlaneRotate': 'PointnetPlaneRotate.PointnetPlusPlaneRotate',
    'pointnetPointPlane': 'PointnetPointPlaneRotate.PointnetPlusPointPlaneRotate',
}


def model_entry(config):
    return sub_model_entry(config, __file__, module_dict)
