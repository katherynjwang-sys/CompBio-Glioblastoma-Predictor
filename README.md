# Glioblastoma MRI Predictor

## Overview
This project explores how to use computation to model and predict glioblastoma tumors using MRI patient data and patient specific brain physiological traits

## Sources
The methods and concepts used in this project are inspired by the following publications:

1. Miller, H. A., Lowengrub, J., & Frieboes, H. B. (2022). *Modeling of Tumor Growth with Input from Patient-Specific Metabolomic Data*. Annals of Biomedical Engineering. https://doi.org/10.1007/s10439-022-02904-5

2. Le, M., Delingette, H., Kalpathy-Cramer, J., Gerstner, E. R., Batchelor, T., Unkelbach, J., & Ayache, N. (2017). *Personalized Radiotherapy Planning Based on a Computational Tumor Growth Model*. IEEE Transactions on Medical Imaging. https://doi.org/10.1109/TMI.2016.2626443

This repository is a reproduction of computational modeling methods used in both papers, based upon two main equations explored in the two

Patient data was acquired from RHUH-GBM Cancer Imaging Archive

## Limitations
Due to lack of computing power, this model can only iterate through the prediction algorithm around 15 times, unlike original models which could iterate around 2000 times. This significantly reduces the accuracy

## Future Work
- Improve model accuracy
- Train on larger datasets
- Add visualization for CSF and Skull Bound (files currently too big to be loaded)

## Disclaimer
This project is intended for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis or actual treatment decisions.
IM NOT RESPONSIBLE FOR POOR DECISIONS

## Other
Collaborative efforts include Aphlas 2 Group at CMU Summer CompBio 2026, Ms. Beautiful, and Dr. RB
