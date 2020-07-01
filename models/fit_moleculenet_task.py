

MOLECULENET_TASK_DEPENDENT_INFORMATION = {
    'Tox21': {'is_regression' : False, 'label_count' : 12},
    'BBBP': {'is_regression' : False, 'label_count' : 1},
    'ToxCast': {'is_regression' : False, 'label_count' : 617},
    'SIDER': {'is_regression' : False, 'label_count' : 27},
    'ClinTox': {'is_regression' : False, 'label_count' : 2},
    'MUV': {'is_regression' : False, 'label_count' : 17},
    'HIV': {'is_regression' : False, 'label_count' : 1},
    'BACE': {'is_regression' : False, 'label_count' : 1},
}


def get_moleculenet_task_dependent_arguments(task):
    return MOLECULENET_TASK_DEPENDENT_INFORMATION[task]