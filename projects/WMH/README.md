White matter hyperintensity segmentation.

This Project Organization

-------------------------
```
├─ data                           <- folder contain link to data [readme](data/README.md)
├
├─ experiments                    <- folder contain `.ipynb` notebooks for experiments
├
├─ utils                          <- folder contain additional code
├──── data_utils.py               <- functions for Data loading, Augmentation, splitting/combining and etc
├──── metrics.py                  <- functions for metric computing
├──── mulptiprocessing.py         <- functions for parallel batch iterating
├──── nn_utils.py                 <- functions for batch iteration
├──── pytorch_utils.py            <- Usefull functions for pytorch
├
├─ reports                        <- folder contain explanation, figures, reports
├──── intro.pdf                   <- Slides about competition
├──── logs
├──── figures                     <- Segmentation results
├
├─ models                         <- folder contain models for experiments (DL architectures)
├──── Unet                        <- some 3d unet like but not
├──── ENet                        <- 2d/3d enet from MaxPisov
├
├─ src                            <- folder contain source code for use in this project.
├──── __init__.py
├──── model                       <- Scripts to train models and then use trained models to make predictions
├─────── features.py
├─────── train.py
├─────── predict.py
├─ docker                         <- Dockerfile and other useful files
├──── Dockerfile
├

```
