# Packages:
- I used anaconda for my environment but also generated a `requirements.txt` file that you can use with and without anaconda
- One could likely run this code without installing all packages in `requirements.txt` by simply installing:
    - numpy, pandas, matplotlib, scikit-learn
    - xgboost
    - geopandas
- Most likely any recent(ish) version of above main packages should work to run this code

# Approach
### Step 1:
- Build a binary classifier using spacial and population features (features that measure population density such as number of people in X mile radius and number of large cities in X mile radius, etc) to predict whether or not a given city is in a metropolitan-like area or not
- Labels:
    - 0: Examples that were labeled `NONMETROPOLITAN AREA` in the dataset
    - 1: Examples that were labeled `MSA` in the dataset
- The classifier in theory should learn what spacial and population properties are associated with what the US government deems to be an MSA

### Step 2:
- For all cities predicted by classifier from step 1 to be in a metropolitan area, cluster the cities into specific MSAs using a custom algorithm made by me from scratch. My approach does **not** use statistical learning or loss minimization. 
    - I had a lot of priors and strong assumptions about how economies form around large cities and how we **should** think of MSAs so it was easier to code my own algorithm than to use a known unsupervised learning algorithm and edit it heavily.
    - Document describing algorithm and implementation details: https://docs.google.com/document/d/1R-KJb4pInab0-0G7-8_LXGGcG9W_yTcjyPCXug9CNqs/edit
    
### Step 3:
- All cities predicted to not be in an MSA are categorized as `countryName__stateName__NONMETROPOLITAN AREA` similarly to how the data given shows non MSAs are simply grouped by state
    - Ex: In data given, you will see `NY NONMETROPOLITAN AREA` representing a city not in the metro area that is inside NY state 

### General notes:
- All MSA groupings begin and end within a country's borders. No MSA extends multiple countries by design. And as seen in the bullet point from Step 3 above, all non metro areas within the same (country, state) are grouped. This was a choice to reflect how an analyst would likely want to group cities into units. I believe it is most natural to structure MSAs to respect country borders due to the potentially large political friction between any 2 random countries. Traveling and working between countries is likely non trivial so its safe to not group cities between nations.

# Data cleaning and exploratory data analysis
- The data is very messy with a substantial amount of errors. My approach was heavily influenced by this fact. The classifier is trained on cleaned data and the clustering method is robust to the data issues below: 
    - 0 population entries very common
    - Many NAN MSA values
    - 3 repeat latitude longitude combinations
    - Cities repeating with different names, neighborhoods of a city being considered "cities", etc
    - Cities very far from an MSA being considered part of that MSA (cities in the west coast USA should not be in NY MSA)
- I trim each MSA such that no MSA has cities that are more than `300 miles` (haversine distance) from the median (latitude, longitude) of that MSA. `200 miles might be appropriate as well`
    - I do **not** do the same with `NONMETROPOLITAN AREAS` given they can span across a large state like Texas. Classifier is trained on trimmed `MSAs` and the **given** `NONMETROPOLITAN AREAS`

# Code notes
- `helper.py` contains all of the relevant functions, classes, and code
- Class `union_find` within `helper.py` which is a class written by me to construct disjoint sets where the representative is always the highest population element. I add path compression optimization and various helper functions
- Follow the notebooks commentary throughout the notebook to understand what is being computed

# Output
- See `output.csv` which is in the format requested. `new_msa` is the column of interest. Entries in that column have form:
    - Cities predicted to not be part of an MSA: `countryName__stateName__NONMETROPOLITAN AREA`
    - Cities predicted to be part of an MSA: `countryName__stateName__largestCityName__MSA`
        - The largest city in a cluster is the representative of a cluster and thus names the cluster

# Training and evaluation:

### Training the classifier:
- The classifier was tuned via grid search cross validation and the model that maximized mean ROC AUC was chosen. The entire tuning process utilized 90% of the MSA vs Non Metropolitan Area data (USA only)
    - Tuned in Google Colab for GPU: https://colab.research.google.com/drive/1cW25sDRNIpK7Z1m2pGTnNO3POOsNaeyM?usp=sharing 

### Evaluating the classifier:
- The model's accuracy, ROC AUC, FPR, and FNR on the test set (10% of all the MSA vs Non Metropolitan Area data) were all comparable or superior to the logistic regression (with L1 regularization) baseline.
- Another pseudo metric tracked how many people were misclassified by totaling the misclassified populations. We see that the XGBoost model has a smaller misplaced population amount as well (see section in notebook where I do test set eval)
    - One could assign a monetary value to the misclassified population of people by assuming some average GDP per capita for rural vs metro areas as well if desired (not done here) 

### Evaluating the clustering:

#### Supervised Metrics:
- As a sanity check, I calculated Adj Rand Index with the actual MSA assignments to show that my clustering system reasonably aligns with the ground truth (as does a baseline DBScan model) [not shown in notebook]

#### Unsupervised Metrics:
- Much of the standard ML metrics to evaluate a clustering mostly fall short. Other clustering algorithms would in theory achieve better silhouette scores due to them optimizing for it directly or indirectly. I argue that the clustering process I present is a **simplified but effective** way that mimics fundamental structures of MSAs. I am **proposing a structure** as opposed to solving for one as ML models generally do. Thus, evaluation was not heavily focused on standard unsupervised ML metrics. The clustering visualizations and spot checks show how an analyst can easily interpret and use these clusterings.


# References:
- https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
- https://stackoverflow.com/questions/34579213/dbscan-for-clustering-of-geographic-location-data
- https://leetcode.com/discuss/general-discussion/1072418/Disjoint-Set-Union-(DSU)Union-Find-A-Complete-Guide
- https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6
- https://en.wikipedia.org/wiki/Metropolitan_statistical_area


# Misc
- Exploratory data analysis and looking into data issues not included in this notebook. Basic analysis of sizes of clusters, plotting, and average distances between cities as well as distances between large population areas were noted
