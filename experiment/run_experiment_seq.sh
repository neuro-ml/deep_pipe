#!/usr/bin/env bash

cd $1
cd experiment_0

for d in ../experiment_*/; do
    cd ${d}
    snakemake || exit 1
done

