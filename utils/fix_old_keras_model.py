import h5py

def fix_old_keras_model(file):
    f = h5py.File(file, mode="r+")
    model_config_string = f.attrs.get("model_config")
    if model_config_string.find('"groups": 1,') != -1:
        model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()
        model_config_string = f.attrs.get("model_config")
        assert model_config_string.find('"groups": 1,') == -1
    f.close()