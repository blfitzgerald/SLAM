# HTCondor Workflow for SLAM

This repository contains the full HTCondor-enabled codebase for the project described in our paper:

**"An L-Moments-Based Hypothesis Test to Identify Homogeneous Storm Transposition Regions"**  
Benjamin FitzGerald, Daniel Wright, Lei Yan, Alyssa Hendricks Dietrich, and Antonia Sebastian 
Submitted to Journal of Hydrology,6/2/2025  
---

## Overview

This project uses a four-stage pipeline implemented with HTCondor to create a transposition domain for a watershed of interest using a gridded precipitation product. Each stage is run as a distinct Condor job with its own `submit` file, executable script, and Python code.
---

## Workflow Structure

The pipeline includes the following stages:

1. **Stage 1 – Point Precipitation to Watershed Averaged Precipitation**  
   Creates a gridded dataset of watershed averaged precipitation by transposing the watershed shape across the whole domain and caluclating precipition across it.

2. **Stage 2 – Annual Maxima Calculation**  
   Calculates the annula maxima for the watershed average precipitation.

3. **Stage 3 – L-Moment Calculation**  
   For each watershed transposition, takes the average of all annual maxima to create a composite then calculates the L-moments of the grid cells in the composites.

4. **Stage 4 – Hypothesis Test and Final Domain Drawing**  
   Performs a hypothesis test comparing the L-moments of the watershed transpositions to the L-moments of the watershed. The a final domain is drawn based on the results. 

Each stage is organized in its own subdirectory under `scripts/` and can be submitted independently via `condor_submit`.

---

## 📁 Repository Layout

