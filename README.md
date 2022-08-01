# AlkaneVirial

Supporting Information for "Machine Learning from Heteroscedastic Data: Second Virial Coefficients of Alkane Isomers" paper.
Authors: Arpit Bansal, Andrew J. Schultz, David A. Kofke and Johannes Hachmann
Journal of Physical Chemistry B (2022)

List of files/folders submitted as part of Supporting Information:
- 2D_Descriptors_after_feature_selection.txt: Lists all the 2D descriptors from Dragon 7.0 remaining after feature selection along with a brief description
- B2_MSMC.txt: Lists 12,671 B2 values for all alkane isomers up to C15 and 1000 each randomly chosen from C16 to C20, computed using Mayer Sampling Monte Carlo (MSMC) simulations in Etomica
- B2_DNN.txt: Lists 618,049 B2 values for all alkane isomers up to C20 (except methane) predicted using the deep neural network (DNN)
- Shape_Descriptors_Alkanes_up_to_C15.txt: Lists shape descriptors generated using Monte Carlo simulations in Etomica for all alkane isomers up to C15. 
- B2_Prediction_DNN.py: Python code to train the DNN and make predictions
- B2 vs Descriptors: Zipped folder containing plots of B2 vs features selected for all simulated alkane isomers
