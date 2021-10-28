# dynaPhenoM
This code is for our submission dynaPhenoM: Dynamic Phenotype Modeling from Longitudinal Patient Records Using Machine Learning

The dynaPhenoM mainly contains two modules: DMTM and T-LCA. 
DMTM is developed to learn compressed representation from longitudinal multimodal clinical events.
T-LCA is developed to derive pregression subphenotypes using well-learned compressed representations from DMTM.

Folder: DMTM code
For DMTM, we provide two types of codes, which correspondent to two coding platform: MATLAB and Python.
MATLAB code can be run only on Windows. Python code can be run on Windows and Linux. 
The data folder have a toy data that helps user to run our method and understand the algorithm.

For MATALB code, please run main_BPlink.m
For Python code, please run DMTM_main.py

Folder: T-LCA code
We provide two examples to play with T-LCA.

main_TLCA_toy_data_with_same_length.py: This ignores the irregular visit time. Actually, this is the original LCA for longitudinal subphenotyping.
main_TLCA_toy_data_with_different_length.py: This considers the irregular visit time. This method is used in dynaPhenoM.

If there is any problems about dynaPhenoM, please feel free to contact with Hao: haz4007@med.cornell.edu
