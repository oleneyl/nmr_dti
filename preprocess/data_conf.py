import os, json
import datetime


def add_data_config(dir_path, train_set_name='train', valid_set_name='valid',
                    data_read_type='line_json', includes_nmr=True, **kwargs):
    # You can pass any arguments to function which want to be tracked.
    config = {
        'build_date': str(datetime.datetime.now()),
        'train_set': train_set_name,
        'valid_set': valid_set_name,
        'data_read_type': data_read_type,
        'includes_nmr': includes_nmr
    }

    for k, v in kwargs.items():
        config[k] = v

    with open(os.path.join(dir_path, 'config.json'), 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
