# DM3-ML : Unsupervised Learning and Dimensionality Reduction 
URL to have access to all the codes : https://github.com/ChristopheYe/DM3-ML.git

I. PROJECT'S TITLE

DM3 ML

II. PROJECT DESCRIPTION

The code is written with Python on different Jupyter Notebook
In addition of many classic libraries like pandas, numpy, time, and matplotlib, I used sklearn libraries :
import numpy as np
import time
import pandas as pd
import io
import matplotlib.pyplot as plt
import mlrose_hiive
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import completeness_score

First part : Experimentation with two clustering algorithms : 
1. K-means clustering
2. Expectation Maximization

Second part : Experimentation with four dimensionality reduction algorithms : 
1. Principal Component Analysis
2. Independant Component Analysis
3. Gaussian Random Projection
4. Decision Tree

All the different clustering and dimensionality reduction algorithms used were taken from the library sklearn.

The 2 initial datasets I work with :
1. wine-quality-white-and-red.csv
2. Movie Dataset.csv

And then I worked with plenty of others :
3. wine_quality_pca.csv
4. wine_quality_ica.csv
5. wine_quality_grp.csv
6. wine_quality_DT.csv
7. wine_quality_Kmeans.csv
8. wine_quality_EM.csv
9. Movie_pca.csv
10. Movie_ica.csv
11. Movie_grp.csv
12. Movie_DT.csv
13. Movie Dataset_Kmeans.csv
14. Movie_EM.csv

Third part : Experimentation with clustering after dimensionality reduction

Fourth part : Neural network after dimensionality reduction on my dataset about 'wine' and compare the performance to a Neural network without dimensionality reduction

Fifth part : Neural Network on a new dataset with clustering feature n my dataset about 'movie' and compare the performance to the original dataset.

III. HOW TO INSTALL AND RUN THE PROJECT

1. Download Anaconda-Navigator
2. Use a Jupyter Notebook
3. Import all the libraries
4. Run all the different codes in the GitHub
