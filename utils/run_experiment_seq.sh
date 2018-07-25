#!/usr/bin/env bash

cd $1
cd experiment_0

for d in ../experiment_*/; do
    cd ${d}
    python -m dpipe run_experiment --config_path ../resources.config || exit 1
done

