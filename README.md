# Machine Learning Project: Handwritten Digit Recognition

**Polytech Nice Sophia**
**Date:** June 10, 2025
**Authors:** Ben Khalifa Emna, Honakoko Giovanni, Ziad Zineb

## 1. Project Overview
This project focuses on **Supervised Learning** to train a machine to recognize handwritten digits. The core objective is to develop an optimized algorithm capable of identifying a digit from an image by learning patterns within a dataset.

## 2. Dataset
We utilized the `load_digits` dataset from the Python `scikit-learn` (sklearn) library.
* **Content:** 1,797 images of handwritten digits.
* **Structure:** The data matrix $X$ has dimensions $(1797, 64)$.
* **Format:** Each data point is an $8 \times 8$ pixel grayscale image, flattened into a vector where coefficients represent pixel intensity ranging from 0 (black) to 16 (white).

## 3. Data Engineering & Preprocessing
To prepare the data for training, we implemented a robust preprocessing pipeline including normalization and feature extraction.

### 3.1 Normalization
We normalized the pixel intensities in matrix $X$ to scale values between $[0, 1]$.

### 3.2 Feature Extraction
We employed three distinct feature extraction techniques to improve model performance:
1.  **Principal Component Analysis (PCA):**
    * Used to reduce dimensions by diagonalizing the correlation matrix.
    * **Analysis:** We found that retaining **30 principal components** captures **95% of the information (variance)**, significantly reducing data flow while maintaining signal integrity.
    * **Global vs. Local:** We explored both Global PCA (on the whole dataset) and Local PCA (on specific classes) to refine reconstruction.
2.  **Image Partitioning:**
    * The $8 \times 8$ images were split into three zones (rows 1-3, 4-5, 6-8) to analyze average gradient variations in specific areas.
3.  **Sobel Filters:**
    * Applied convolution filters to detect localized gradient variations, specifically identifying vertical and horizontal contours.

### 3.3 The Pipeline
We automated the process using a `sklearn` **Pipeline** object, which chains the scaling, feature union, and the ML algorithm into a streamlined sequence.

## 4. Methodology
The dataset was split into **training** and **test** sets. We verified that the distribution of classes remained quasi-uniform across both sets using the **Bhattacharyya distance** (distance $\approx 10^{-4}$).

We developed and compared two primary models:
* **Model 1:** Standard pipeline with local preprocessing.
* **Model 2:** Similar to Model 1 but includes a **local PCA feature**. This involved creating sub-databases for each class to reduce noise and improve recognition.

The classification strategy relies on linear separation, utilizing concepts like **OvO** (One-versus-One) and **OvR** (One-versus-Rest).

## 5. Results
The models were evaluated using confusion matrices and accuracy scores. **Model 2 (Optimized)** yielded the best performance.

| Metric | Model 2 (Non-Optimized) | Model 2 (Optimized) |
| :--- | :--- | :--- |
| **Training Accuracy** | 100% | **100%** |
| **Test Accuracy** | 97.59% | **98.89%** |
| **Correct Predictions** | 527 / 540 | **534 / 540** |

The confusion matrix for the optimized model demonstrates distinct diagonal dominance, indicating high classification success rates across all digits.

## 6. Conclusion
While Neural Networks were considered and noted for their ease of implementation, our optimized Machine Learning model (Model 2) proved to be more precise given the time constraints of this project. The combination of PCA, zone partitioning, and Sobel filters provided a robust mechanism for digit recognition.
