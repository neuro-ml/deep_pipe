#!/usr/bin/env bash

jupyter nbconvert --to rst tutorials/*.ipynb
sed -i 's/code:: ipython3/code-block:: python3/g' tutorials/*.rst
sed -i 's/\$/\`/g' tutorials/*.rst
make clean html
