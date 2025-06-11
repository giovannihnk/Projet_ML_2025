# Split the data 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# Load processed feature matrix and labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# TODO: Add any util functions you may have from the previous script
digits = load_digits()

# TODO: Load the raw data
X,y = digits.data, digits.target
X=X/16
#####
#In machine learning, we must train the model on one subset of data and test it on another.
#This prevents the model from memorizing the data and instead helps it generalize to unseen examples.
#The dataset is typically divided into:
#Training set → Used for model learning.
#Testing set → Used for evaluating model accuracy.
# The training set is also split as a training set and validation set for hyper-parameter tunning. This is done later
#
# Split dataset into training & testing sets


##########################################
## Train/test split and distributions
##########################################


# 1- Split dataset into training & testing sets
# TODO: FILL OUT THE CORRECT SPLITTING HERE
X_train, X_test, y_train, y_test = X[0:1437,:], X[1437:,:], y[0:1437], y[1437:]
### If you want, you could save the data, this would be a good way to test your final script in the same evaluation mode as what we will be doing
np.save("X_train.npy", X_train)
np.save("test_data.npy", X_test)
np.save("y_train.npy", y_train)
np.save("test_label.npy", y_test)
####

# TODO: Print dataset split summary...
def get_statistics_text(targets):
    classes , counts = np.unique(targets, return_counts=True)
    return classes, counts
get_statistics_text(digits.target)


# Création des DataFrames
pixel_columns = [f'pixel_{i}' for i in range(X.shape[1])]
X_train_df = pd.DataFrame(X_train, columns=pixel_columns)
X_test_df = pd.DataFrame(X_test, columns=pixel_columns)
y_train_df = pd.DataFrame(y_train, columns=['label'])
y_test_df = pd.DataFrame(y_test, columns=['label'])

# Ajouter les étiquettes aux features sur les targets et les data
X_train_df['y_train'] = y_train
X_test_df['y_test'] = y_test

# Summary du train et du test
print("Résumé de X_train :")
print(X_train_df.describe())
print(f"Feature matrix X_train_df shape: {X_train_df.shape}. Max value = {np.max(X_train_df)}, Min value = {np.min(X_train_df)}, Mean value = {np.mean(X_train_df)}")

print("\nRésumé de X_test :")
print(X_test_df.describe())
print(f"Feature matrix X_test_df shape: {X_test_df.shape}. Max value = {np.max(X_test_df)}, Min value = {np.min(X_test_df)}, Mean value = {np.mean(X_test_df)}")

print("\nRépartition des classes dans y_train :")
print(y_train_df['label'].value_counts())
print(f"Feature matrix y_train_df shape: {y_train_df.shape}. Max value = {np.max(y_train_df)}, Min value = {np.min(y_train_df)}, Mean value = {np.mean(y_train_df)}")

print("\nRépartition des classes dans y_test :")
print(y_test_df['label'].value_counts())
print(f"Feature matrix X_train_df shape: {y_test_df.shape}. Max value = {np.max(y_test_df)}, Min value = {np.min(y_test_df)}, Mean value = {np.mean(y_test_df)}")


# TODO: ... and plot graphs of the three distributions in a readable and useful manner (bar graph, either side by side, or with some transparancy)
classes, counts = np.unique(y, return_counts=True)
classes_train, counts_train = np.unique(y_train, return_counts=True)
classes_test, counts_test = np.unique(y_test, return_counts=True)
# Largeur de chaque barre
width = 0.25

# Positions sur l'axe x
x = np.arange(len(classes))
plt.figure(figsize=(10,5))
plt.bar(x - width, counts/183, width=width, color='skyblue', label='whole dataset')
plt.bar(x, counts_train/146, width=width, color='lightgreen', label='training set')
plt.bar(x + width, counts_test/37, width=width, color='lightpink', label='test set')

plt.xlabel('Classes')
plt.ylabel("Fréquence")
plt.title("Distribution des classes")
plt.xticks(x, classes)  # Affiche les vraies classes sur l'axe x
plt.legend(title='Dataset')
plt.show()

# Distance de Bhattacharyya, vient affirmer le graphe précédent
random.seed(10)

def bhattacharyya_distance(p, q, epsilon=1e-10):
    p /= p.sum()
    q /= q.sum()
    bc = np.sum(np.sqrt(p * q))
    # Distance
    return -np.log(bc)

# Comptes de classes
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)

# Distributions (probabilités)
train_dist = train_counts / train_counts.sum()
test_dist = test_counts / test_counts.sum()

distance = bhattacharyya_distance(train_dist, test_dist)
print(f"Distance de Bhattacharyya entre les distributions de y_train et y_test : {distance:.5f}")


# TODO: (once the learning has started, and to be documented in your report) - Impact: Changing test_size affects model training & evaluation.


##########################################
## Prepare preprocessing pipeline
##########################################

# We are trying to combine some global features fitted from the training set
# together with some hand-computed features.
# 
# The PCA shall not be fitted using the test set. 
# The handmade features are computed independently from the PCA
# We therefore need to concatenate the PCA computed features with the zonal and 
# edge features. 
# This is done with the FeatureUnion class of sklearn and then combining everything in
# a Pipeline.
# 
# All elements included in the FeatureUnion and Pipeline shall have at the very least a
# .fit and .transform method. 
#
# Check this documentation to understand how to work with these things 
# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py

# Example of wrapper for adding a new feature to the feature matrix
from sklearn.base import BaseEstimator, TransformerMixin

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
        return sobel_feature

# TODO: Fill out the useful code for this class
class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        zone_features = []
        for img in X:
            img_reshaped = img.reshape(8, 8)
            zone1 = np.mean(img_reshaped[0:4, :])  # lignes 0 à 2
            zone2 = np.mean(img_reshaped[4:6, :])  # lignes 3 à 4
            zone3 = np.mean(img_reshaped[6:9, :])  # lignes 5 à 7
            zone_features.append([zone1, zone2, zone3])
        return np.array(zone_features)

# TODO: Create a single sklearn object handling the computation of all features in parallel
all_features = PCA(n_components=25)  #feature union

F = all_features.fit(X_train,y_train).transform(X_train)
# Let's make sure we have the number of dimensions that we expect!
print("Nb features computed: ", F.shape[1])

# Now combine everything in a Pipeline
# The clf variable is the one which plays the role of the learning algorithms
# The Pipeline simply allows to include the data preparation step into it, to 
# avoid forgetting a scaling, or a feature, or ...

Features = FeatureUnion([("pca", F), ("zonal features",ZonalInfoPreprocessing()), ("edges features",EdgeInfoPreprocessing() )])  #concatenation des 3 features

# TODO: Write your own pipeline, with a linear SVC classifier as the prediction
clf = Pipeline([("features", all_features),("scaler", StandardScaler()), ("classifier", SVC())])


##########################################
## Premier entrainement d'un SVC
##########################################

# TODO: Train your model via the pipeline

clf.fit(X_train,y_train)

# TODO: Predict the outcome of the learned algorithm on the train set and then on the test set 
predict_test = clf.predict(X_test)
predict_train = clf.predict(X_train)

print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))

# TODO: Look at confusion matrices from sklearn.metrics and 
# 1. Display a print of it
# 2. Display a nice figure of it
# 3. Report on how you understand the results


# TODO: Work out the following questions (you may also use the score function from the classifier)
print("\n Question: How does changing test_size influence accuracy?")
print("Try different values like 0.1, 0.3, etc., and compare results.\n")



##########################################
## Hyper parameter tuning and CV
##########################################
# TODO: Change from the linear classifier to an rbf kernel
# TODO: List all interesting parameters you may want to adapt from your preprocessing and algorithm pipeline
# TODO: Create a dictionary with all the parameters to be adapted and the ranges to be tested

# TODO: Use a GridSearchCV on 5 folds to optimize the hyper parameters
grid_search = GridSearchCV(DummyClassifier()) #, verbose=10)
# TODO: fit the grid search CV and 
# 1. Check the results
# 2. Update the original pipeline (or create a new one) with all the optimized hyper parameters
# 3. Retrain on the whol train set, and evaluate on the test set
# 4. Answer the questions below and report on your findings

print(" K-Fold Cross-Validation Results:")
print(f"- Best Cross-validation score: {grid_search.best_score_}")
print(f"- Best parameters found: {grid_search.best_estimator_}")
#####
print("\n Question: What happens if we change K from 5 to 10?")
print("Test different K values and compare the accuracy variation.\n")


##########################################
## OvO and OvR
##########################################
# TODO: Using the best found classifier, analyse the impact of one vs one versus one vs all strategies
# Analyse in terms of time performance and accuracy


# Print OvO results
print(" One-vs-One (OvO) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print("- Impact: Suitable for small datasets but increases complexity.")

print("\n Question: How does OvO compare to OvR in execution time?")
print("Try timing both methods and analyzing efficiency.\n")
###################
# TODO:  One-vs-Rest (OvR) Classification


# Print OvR results
print(" One-vs-Rest (OvR) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print("- Impact: Better for large datasets but less optimal for highly imbalanced data.")

print("\n Question: When would OvR be better than OvO?")
print("Analyze different datasets and choose the best approach!\n")
########



