# PCA_CAPs

 This repository contains the example scripts to run the PCA CAPs analyses (without the GUI) for one and two populations.
 For concerns, question and feedback, please contact: Samantha Weber, samantha.weber@unibe.ch // samantha.weber@bli.uzh.ch

 v1.0 June 2022

## Data Preparation
 Please make sure to prepare your data accordingly as reported in the original toolbox paper Bolton et al., TbCAPs: A toolbox for co-activation pattern analysis. NeuroImage 2020.

## Citation

 If you use this script please cite:  
- Weber et al., "Transient resting-state salience-limbic co-activation patterns in functional neurological disorders", NeuroImage:Clinical, 2024, doi: 10.1016/j.nicl.2024.103583
- Bolton et al., TbCAPs: A toolbox for co-activation pattern analysis. NeuroImage 2020. doi: 10.1016/j.neuroimage.2020.116621 

## Usage: 

1. Two Population: 
- Run "Script_twopop_PCA_CAP_SW.m
- Follow the instructions in the script
- This script calculates CAPs based on the reference population (group 1). 

2. One Population: 
- Run "Script_onepop_PCA_CAP_SW.m
- Follow the instructions in the script
- This scripts calculates CAPs based on one reference population. This "reference" population can - if necessary - contain different groups. Group statistics can still be done based on the temporal metrics. 

3. Visualization: 
- in order to visualize your CAPs you can type CAP_SW in your matlab command window and load the output file from step (1)/(2).
- For proper visualization I recommend using other toolboxes e.g., fsleyes or xjview .

## Credits

Code written by Samantha Weber with subfunctions previously published by Thomas Bolton on https://c4science.ch/source/CAP_Toolbox/
