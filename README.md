# MAPKAPK2 Inhibition Prediction Ensemble Deep Learning Implementation

The ten final individual models can be found under trained_models.

All compound sets and data used for training, testing, and evaluation can be found under data.

Code to train all models is located in the model_generation folder, containing two subfolders for models using RDKit features and molfeat features.

The implementation of the ensemble comprising the ten models under trained_models is shown in evaluation.py, where the unknown compound set for evaluation can be replaced, adjusting file formatting accordingly.

Files related to molecular docking are in the docking folder.