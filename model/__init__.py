def get_model(modelName):
    if modelName == 'LightGCL':
        from .LightGCL import LightGCL
        return LightGCL
    else:
        raise NotImplementedError('Network {} is not implemented'.format(modelName))