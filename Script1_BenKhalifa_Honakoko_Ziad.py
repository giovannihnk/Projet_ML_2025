import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import seaborn as sns
from sklearn.decomposition import PCA
from skimage import data
from skimage import filters
from sklearn.preprocessing import StandardScaler


##########################################
## Data loading and first visualisation
##########################################

# Load the handwritten digits dataset
digits = load_digits() 

# Graph of the first 4 images from the data base 
fig = plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(digits.images[i], cmap="binary")
    plt.axis('off')
plt.show()


# Display at least one random sample par class (some repetitions of class... oh well)
def plot_multi(data, y):
    '''Plots 16 digits'''
    nplots = 16
    nb_classes = len(np.unique(y))
    cur_class = 0
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        to_display_idx = np.random.choice(np.where(y == cur_class)[0])
        plt.imshow(data[to_display_idx].reshape((8,8)), cmap='binary')
        plt.title(cur_class)
        plt.axis('off')
        cur_class = (cur_class + 1) % nb_classes
    plt.show()


plot_multi(digits.data, digits.target)

##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(targets):
    # return Label names and number of elements per class
    return np.unique(targets, return_counts=True)
    


# Call the previous function and generate graphs and prints for exploring and visualising the database
classes, counts = get_statistics_text(digits.target)
print(f"classes : {classes}, number of elements per classes : {counts}")
plt.bar(classes,counts,color='skyblue')
plt.xlabel('Classes')
plt.ylabel("fréquence")
plt.title("distribution des classes")
plt.show()

##########################################
## Start data preprocessing
##########################################

# Access the whole dataset as a matrix where each row is an individual (an image in our case) 
# and each column is a feature (a pixel intensity in our case)
## X = [
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 1 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 2 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 3 as a row
#  [Pixel1, Pixel2, ..., Pixel64]   # Image 4 as a row
#]

# Create a feature matrix and a vector of labels
X = digits.data 
y = digits.target 

# Print dataset shape
print(f"Feature matrix shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")


# Normalize pixel values to range [0,1]
F = X/16 

# Print matrix shape
print(f"Feature matrix F shape: {F.shape}. Max value = {np.max(F)}, Min value = {np.min(F)}, Mean value = {np.mean(F)}")

##########################################
## Dimensionality reduction
##########################################


### just an example to test, for various number of PCs
sample_index = 0
original_image = F[sample_index].reshape(8, 8)  # Reshape back to 8×8 for visualization

# Using the specific sample above, iterate the following:
# * Generate a PCA model with a certain value of principal components
# * Compute the approximation of the sample with this PCA model
# * Reconstruct a 64 dimensional vector from the reduced dimensional PCA space
# * Reshape the resulting approximation as an 8x8 matrix
# * Quantify the error in the approximation
# Finally: plot the original image and the 15 approximation on a 4x4 subfigure
# Assuming X is the original data matrix

pca = PCA(n_components=15)
pca.fit(F)

X_pca = pca.transform(F[sample_index].reshape(1,-1)) # projection sur la base
print("x_pca : \n",X_pca)
print(X_pca.shape)


X_inv = pca.inverse_transform(X_pca)  # Reprojection dans l’espace initial 
reconstructed_image = X_inv.reshape(8, 8)

plt.subplot(1,2,1)
plt.imshow(original_image, cmap='binary')
plt.title("image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed_image, cmap='binary')
plt.title("image reconstruite")
plt.axis("off")
plt.show()

# visualition of pca in 2D

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)  

# Display of projected points
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection PCA en 2D")
plt.colorbar(scatter, ticks=range(10), label="Classe")
plt.grid(True)
plt.show()


### display of images that correspond to the eigenvectors ###

matrice_passage = pca.components_
# print("passage",matrice_passage.shape)
for i in range(15):
    plt.subplot(3,5,i+1)
    matrice = matrice_passage[i,:].reshape(8,8)
    plt.imshow(matrice,cmap="binary")
    plt.axis("off")
plt.show()


#### Expolore the explanined variance of PCA and plot 

# PCA from scratch
X_centered = F - np.mean(F, axis=0)
cov_matrix = np.transpose(X_centered) @ X_centered # : matrice de corrélation
# Décomposition en valeurs propres
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# Trier les valeurs/vecteurs propres par ordre décroissant
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Display of eigenvalues
plt.figure(figsize=(10,5))
plt.plot(eigenvalues, marker='o')
plt.title("Variance par composante principale")
plt.xlabel("Indice de la composante")
plt.ylabel("Valeur propre (variance expliquée)")
plt.grid(True)
plt.show()
# Display of the cumulative variance
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
plt.figure(figsize=(10,5))
plt.plot(cumulative_variance, marker='o')
plt.title("Variance cumulée")
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulée (%)")
plt.grid(True)
plt.axhline(0.95, color='red', linestyle='--', label='95% variance')
plt.legend()
plt.show()



### Display the whole database in 2D: 
database = []
for i in range(F.shape[0]):
    database.append(F[i].reshape(8,8))

print(database[0])

### Creation of a 20 dimensional PCA-based feature matrix

pca_20 = PCA(n_components=20)
pca_20.fit(F)
F_pca = pca_20.transform(F)

# Print reduced feature matrix shape
print(f"Feature matrix F_pca shape: {F_pca.shape}")


##########################################
## Feature engineering
##########################################
### # Function to extract zone-based features
###  Zone-Based Partitioning is a feature extraction method
### that helps break down an image into smaller meaningful regions to analyze specific patterns.
def extract_zone_features(images):
    '''Break down an 8x8 image in 3 zones: row 1-3, 4-5, and 6-8'''
    zone_features = []
    for img in images:
        img_reshaped = img.reshape(8, 8)
        zone1 = np.mean(img_reshaped[0:4, :])  # lignes 0 à 2
        zone2 = np.mean(img_reshaped[4:6, :])  # lignes 3 à 4
        zone3 = np.mean(img_reshaped[6:9, :])  # lignes 5 à 7
        zone_features.append([zone1, zone2, zone3])
    return np.array(zone_features)

# Apply zone-based feature extraction
F_zones = extract_zone_features(F)

# Print extracted feature shape
print(f"Feature matrix F_zones shape: {F_zones.shape}")


### Edge detection features

## Get used to the Sobel filter by applying it to an image and displaying both the original image 
# and the result of applying the Sobel filter side by side
# Compute the average edge intensity for each image and return it as an n by 1 arra
edges = []
for img in F:
    edges.append(np.mean(filters.sobel(img)))
F_edges = np.array(edges)
F_edges = F_edges.reshape(-1, 1) 

# Print feature shape after edge extraction
print(f"Feature matrix F_edges shape: {F_edges.shape}")

### connect all the features together

# TODO: Concatenate PCA, zone-based, and edge features
F_final = np.concatenate((F_pca,F_zones,F_edges),axis=1)

# TODO: Normalize final features
s = StandardScaler()
F_final = s.fit_transform(F_final)


# Print final feature matrix shape
print(f"Final feature matrix F_final shape: {F_final.shape}")


### function to seperate one label from the whole dataset
def GetImages(c,target,database):
    pos = np.where(np.array(target) == c)[0]
    label = []
    for i in pos:
        label.append(database[i])
    return np.array(label)


### Exemple with label 0
Label0 = GetImages(0,digits.target,F)
original_image0 = Label0[sample_index].reshape(8, 8)  # Reshape back to 8×8 for visualization


pca_0 = PCA(n_components=15)
pca_0.fit(Label0)

X_pca_0 = pca_0.transform(Label0[sample_index].reshape(1,-1)) # projection sur la base 
print("x_pca : \n",X_pca_0)
print(X_pca_0.shape)


X_inv_0 = pca_0.inverse_transform(X_pca_0)  # Reprojection dans l’espace initial 
reconstructed_image0 = X_inv_0.reshape(8, 8)

plt.subplot(1,2,1)
plt.imshow(original_image0, cmap='binary')
plt.title("image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed_image0, cmap='binary')
plt.title("image reconstruite")
plt.axis("off")
plt.show()