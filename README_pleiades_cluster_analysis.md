# Pleiades (M45) Cluster Analysis with Unsupervised Learning

> Identifying open-cluster member stars from Gaia DR3 astrometry using K-means, Gaussian Mixture Models, and Principal Component Analysis.

## Overview

Real-world unsupervised-learning project on Gaia DR3 data toward the Pleiades open cluster (M45). The cluster members move coherently through the galaxy, while foreground and background stars move randomly — meaning the cluster is identifiable as a tight overdensity in proper-motion space. This project separates cluster from field stars, then characterises the cluster with PCA and a Hertzsprung-Russell diagram.

## Key Features

- **Quality cuts on real astronomical data** — parallax sign, relative parallax error, and physical motion bounds
- **Two clustering approaches compared** — K-means (hard assignment) and Gaussian Mixture Models (soft, with covariance)
- **Model-selection sweep** — GMM with 2, 3, and 5 components to find the structure that best separates cluster from field
- **Dimensionality reduction with PCA** — full feature set reduced to dominant components, with loadings interpretation
- **Hertzsprung-Russell diagram** as the final domain-science output

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib`

## Approach

After loading the Gaia query, the data is filtered to keep only stars with positive parallaxes, relative parallax error below 20%, and proper motions within physically plausible bounds. This removes spurious matches and very distant background stars whose astrometry isn't reliable.

The filtered sample is then clustered in proper-motion space (`pmra`, `pmdec`), where the Pleiades members appear as a tight cluster offset from the diffuse field-star distribution. K-means with two clusters gives a quick first separation; GMM is used to capture the elongated, non-spherical shape of the cluster more accurately. A five-component GMM resolves additional substructure in the field.

The most concentrated cluster is then standardised and run through PCA across the full Gaia feature set (position, parallax, motion, photometry). The principal components reveal which features carry the most variance, and projection onto PC1–PC2 shows the cluster's intrinsic structure. The final HR diagram (Gaia BP–RP colour vs G-band magnitude) is the standard astrophysical sanity check — Pleiades members should lie along a clean main sequence with a turnoff consistent with the cluster's age.

## Results

[Add a screenshot of the HR diagram here — it's the most striking figure.]

The pipeline outputs:
- Pre- and post-filter proper-motion scatter plots
- K-means and GMM cluster assignments overlaid on proper-motion space
- Cluster centres and member counts printed to stdout
- PCA explained-variance plot and top feature loadings
- Final HR-like diagram for the identified cluster

## How to Run

```bash
git clone https://github.com/<your-username>/pleiades-cluster-analysis.git
cd pleiades-cluster-analysis
pip install -r requirements.txt
```

The cleaned Gaia CSV should be placed at `data/m45-clean.csv`. Then:

```bash
python pleiades_cluster_analysis.py
```

## Data

The input CSV is a cleaned Gaia DR3 query targeting the Pleiades field. Similar queries can be reproduced from the [Gaia Archive](https://gea.esac.esa.int/archive/) using a cone search around RA 56.75°, Dec 24.12°.
