

# HTCondor Workflow for SLAM

This repository contains the full HTCondor-enabled codebase for the project described in our paper:

**"An L-Moments-Based Hypothesis Test to Identify Homogeneous Storm Transposition Regions"**  
Benjamin FitzGerald, Daniel Wright, Lei Yan, Alyssa Hendricks Dietrich, and Antonia Sebastian  
Submitted to *Journal of Hydrology*, June 2, 2025  

---

## Overview

This project uses a four-stage pipeline implemented with HTCondor to create a transposition domain for a watershed of interest using a gridded precipitation dataset. Each stage is run as a distinct Condor job with its own input file, `submit` file, executable script, and Python code.

---

## Workflow Structure

The pipeline includes the following stages:

1. **Stage 1 – Point Precipitation to Watershed-Averaged Precipitation (1.PP2WAP)**  
   Transposes the watershed shape across the full domain and computes watershed-averaged precipitation at each location to create a gridded dataset.

2. **Stage 2 – Annual Maxima Calculation (2.AMC)**  
   Calculates annual maxima from the watershed-averaged precipitation data.

3. **Stage 3 – L-Moment Calculation (3.LMC)**  
   For each transposition, computes the composite of annual maxima and calculates L-moments for each grid cell in the composite.

4. **Stage 4 – Hypothesis Test and Final Domain Drawing((4.HTDD)**  
   Performs a hypothesis test comparing the L-moments of transpositions to the original watershed. A final domain is drawn based on statistically similar regions.

Each stage is organized in its own subdirectory under `scripts/` and can be submitted independently using `condor_submit`.

---
