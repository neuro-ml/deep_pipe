#!/usr/bin/env bash
set -e

cd $1
cd experiment_0

for d in ../experiment_*/; do
    cd ${d}
    snakemake
done

