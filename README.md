# Conceptual Counterfactuals
This repository contains code for the ICML 2022 submission: "Meaningfully Debugging Model Mistakes Using Conceptual Counterfactual Explanations"

## Metadataset Experiments:
1- To reproduce the results associated with the 20 Metadataset scenarios, please have a look at the `evaluate_cce.py` script. <br>
2- Additionally, the notebook in `metadataset-notebooks/Figure2-Metadataset.ipynb` render the Figure 2 in the paper.  <br> 
3- `metadataset-notebooks/Figure3-LowLevel.ipynb` generates the results related to the low-level perturbations.

## Dermatology Experiment:
Code associated with the dermatology experiments can be found in the `dermatology/` folder.  <br>

1- Run the `dermatology/Learn Dermatology Concepts.ipynb` notebook to learn the clinically relevant concepts. <br>
2- Run the `dermatology/Fitzpatrick17k-training.ipynb` notebook to train models on the Fitzpatrick17k dataset. For more information about the dataset and gaining access, please refer to https://github.com/mattgroh/fitzpatrick17k.  <br>
3- Run the `dermatology/Fitzpatrick17k-Evaluation.ipynb` notebook to generate the quantitative results related to the skin type result reported in the paper.

## Cardiology Experiment:
Code associated with the cardiology experiments can be found in the `cxr/` folder.  <br>

1- Check out the `cxr/Learn XR Concepts.ipynb` notebook to learn the clinically relevant concepts. <br>
2- Run the `cxr/Evaluate CXR.ipynb` notebook to generat the quantitative results reported in the paper.  <br>
3- To get access to the SHC dataset, please check out https://stanfordmlgroup.github.io/competitions/chexpert/. To obtain the NIH dataset, please see https://nihcc.app.box.com/v/ChestXray-NIHCC.

## Concept Bank
In `banks/resnet18_bank.pkl`, you can find a concept bank we use with ResNet18, which contains the concept vectors and precomputed margin statistics.