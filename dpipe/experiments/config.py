from . import flat

splitter_name2build_experiment = {
    'cv_111': flat.build,
    'group_cv_111': flat.build,
}

experiment_name2build_experiment = {
    'msegm_predict': flat.build
}
