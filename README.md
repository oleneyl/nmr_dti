NMR_infer
============


# Dependency

- nmrglue
- numpy
- tqdm
- sentencepieece
- requests
- pandas
- pubchempy

# Structure

- data_utils/
  
  - Data preparation (preprocessing) and result will be on `disk`.
  
- data_loader/
  
  - Data loading support in `runtime`.


# Data preparation

- `python3 prepare_data.py --task create_all --conf [CONF_FILE_LOC]`
- `python3 prepare_data.py --task create_join --conf [CONF_FILE_LOC]`