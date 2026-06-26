# Deep Dive into AI: Machine Learning Foundation & Production

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

A comprehensive, hands-on repository mapping machine learning models from mathematical/conceptual foundations to production-ready implementations. The project is organized into structured modules covering both **Supervised** and **Unsupervised** learning at different stages of complexity and engineering levels.

---

## 📂 Repository Structure

```directory
Deep dive into AI/
├── Deep Learning Foundation Class/
│   ├── Backpropagation_triton.py
│   ├── multiple_perceptron_triton.py
│   ├── perceptron.ipynb
│   └── single_perceptron_triton.py
├── Deep Learning Production Class/
│   └── Image Classifier.ipynb
├── Supervised Learning Foundation Class/
│   ├── CART.ipynb
│   ├── Decision Tree.ipynb
│   ├── Linear Regression.ipynb
│   ├── Logistic Regression.ipynb
│   └── Support Vector Machine.ipynb
├── Supervised Learning Production Class/
│   ├── Adaboost.ipynb
│   ├── Gradient boosting.ipynb
│   ├── Production Boosting.ipynb
│   ├── Random Forest.ipynb
│   └── stacking.ipynb
├── Unsupervised Learning Foundation Class/
│   ├── DBSCAN.ipynb
│   ├── Preprocessing.ipynb
│   └── k-means clustering.ipynb
└── Unsupervised Learning Production Class/
    ├── GMM.ipynb
    ├── HDBSCAN.ipynb
    ├── Image Segmentation.ipynb
    ├── PCA.ipynb
    └── kmeans SSL.ipynb
```

---

## 🛠️ Detailed Module Directory

### 1. Supervised Learning Foundation Class
Focuses on the core concepts, internal mechanisms, and math behind fundamental supervised algorithms using standard libraries.

*   **[CART.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Foundation%20Class/CART.ipynb)**
    *   **Core Concepts:** Classification and Regression Trees (CART), Gini Impurity, split conditions.
    *   **Description:** Implements a deep structural inspection of decision trees, traversing the internal split nodes and terminal leaves to print Gini scores, sample distributions, and prediction thresholds.
    *   **Key Libraries:** `sklearn.tree.DecisionTreeClassifier`, `numpy`.
*   **[Decision Tree.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Foundation%20Class/Decision%20Tree.ipynb)**
    *   **Core Concepts:** Multi-class classification (on Iris dataset), regression trees, feature importance.
    *   **Description:** Builds decision trees for classification and regression tasks, exports human-readable tree flowchart rules (`export_text`), and visualizes how splits relate to variable importances.
    *   **Key Libraries:** `sklearn.tree.DecisionTreeClassifier`, `sklearn.tree.DecisionTreeRegressor`, `matplotlib.pyplot`.
*   **[Linear Regression.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Foundation%20Class/Linear%20Regression.ipynb)**
    *   **Core Concepts:** Ordinary Least Squares (OLS) regression, Gradient Descent optimization, feature scaling.
    *   **Description:** Explores simple linear regression (e.g. predicting student marks based on study hours) and multi-variable regression optimized via Stochastic Gradient Descent (SGD) with feature scaling.
    *   **Key Libraries:** `sklearn.linear_model.LinearRegression`, `sklearn.linear_model.SGDRegressor`, `sklearn.preprocessing.StandardScaler`, `matplotlib.pyplot`.
*   **[Logistic Regression.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Foundation%20Class/Logistic%20Regression.ipynb)**
    *   **Core Concepts:** Binary and multi-class logistic classification, probability boundaries.
    *   **Description:** Models class probabilities on the Iris dataset to evaluate decision boundaries and classification metrics.
    *   **Key Libraries:** `sklearn.linear_model.LogisticRegression`, `matplotlib.pyplot`.
*   **[Support Vector Machine.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Foundation%20Class/Support%20Vector%20Machine.ipynb)**
    *   **Core Concepts:** Support Vector Classifiers (SVC), kernel trick (RBF/linear), non-linear decision boundaries.
    *   **Description:** Focuses on separating non-linearly separable synthetic data (like moons or concentric circles) using kernelized SVMs.
    *   **Key Libraries:** `sklearn.svm.SVC`, `sklearn.preprocessing.StandardScaler`, `matplotlib.pyplot`.

---

### 2. Supervised Learning Production Class
Introduces state-of-the-art ensemble methods, gradient boosting architectures, and production-grade optimization libraries.

*   **[Adaboost.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Production%20Class/Adaboost.ipynb)**
    *   **Core Concepts:** Adaptive Boosting (AdaBoost), base estimators (decision stumps), sample reweighting.
    *   **Description:** Demonstrates the sequential ensemble approach of AdaBoost on synthetic datasets (`make_moons`) to build strong classifiers.
    *   **Key Libraries:** `sklearn.ensemble.AdaBoostClassifier`, `sklearn.tree.DecisionTreeClassifier`.
*   **[Gradient boosting.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Production%20Class/Gradient%20boosting.ipynb)**
    *   **Core Concepts:** Gradient Boosted Decision Trees (GBDT), residual fitting, shrinkage.
    *   **Description:** Implements GBDT regression models to iteratively fit decision tree base-learners to the computed gradients/residuals of preceding steps.
    *   **Key Libraries:** `sklearn.ensemble.GradientBoostingRegressor`, `matplotlib.pyplot`.
*   **[Production Boosting.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Production%20Class/Production%20Boosting.ipynb)**
    *   **Core Concepts:** High-performance GBDT frameworks, categorical feature support, hyperparameter tuning.
    *   **Description:** A comparative walkthrough of industrial-grade boosting implementations—**XGBoost**, **LightGBM**, and **CatBoost**—evaluating training speed, performance, and API syntax.
    *   **Key Libraries:** `xgboost`, `lightgbm`, `catboost`, `pandas`, `sklearn.preprocessing.LabelEncoder`.
*   **[Random Forest.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Production%20Class/Random%20Forest.ipynb)**
    *   **Core Concepts:** Bagging, bootstrap aggregating, feature subspace sampling, monotonic constraints, class balancing.
    *   **Description:** Builds random forest ensembles and details key parameters like `max_samples`, `max_features`, custom class weights, and enforcing monotonic relationships.
    *   **Key Libraries:** `sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.BaggingClassifier`.
*   **[stacking.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Supervised%20Learning%20Production%20Class/stacking.ipynb)**
    *   **Core Concepts:** Multi-model stacking, meta-estimators, out-of-fold predictions.
    *   **Description:** Implements stacking architectures combining diverse base models (e.g. Random Forest, SVM, LightGBM) with a meta-classifier (Logistic Regression) to squeeze out maximum accuracy.
    *   **Key Libraries:** `sklearn.ensemble.StackingClassifier`, `sklearn.pipeline.make_pipeline`, `lightgbm`.

---

### 3. Unsupervised Learning Foundation Class
Covers basic clustering paradigms, preprocessing pipelines, and core spatial concepts.

*   **[k-means clustering.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Foundation%20Class/k-means%20clustering.ipynb)**
    *   **Core Concepts:** Centroid-based clustering, inertia, convergence, scaling.
    *   **Description:** Demonstrates standard K-Means clustering on synthetic datasets, highlighting the necessity of proper scaling and how to initialize centroids.
    *   **Key Libraries:** `sklearn.cluster.KMeans`, `sklearn.preprocessing.StandardScaler`, `matplotlib.pyplot`.
*   **[DBSCAN.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Foundation%20Class/DBSCAN.ipynb)**
    *   **Core Concepts:** Density-based clustering, noise detection, core/border points, epsilon optimization.
    *   **Description:** Cluster data of arbitrary shapes (e.g., moons) using DBSCAN and demonstrates how to choose the optimal `eps` using k-distance graphs via Nearest Neighbors.
    *   **Key Libraries:** `sklearn.cluster.DBSCAN`, `sklearn.neighbors.NearestNeighbors`, `matplotlib.pyplot`.
*   **[Preprocessing.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Foundation%20Class/Preprocessing.ipynb)**
    *   **Core Concepts:** Pipeline integration, grid-searching, clustering as dimensionality reduction/representation.
    *   **Description:** Builds ML pipelines where K-Means is used as a preprocessing/feature-extraction step before a Logistic Regression classifier, optimized via GridSearchCV.
    *   **Key Libraries:** `sklearn.cluster.KMeans`, `sklearn.pipeline.Pipeline`, `sklearn.model_selection.GridSearchCV`.

---

### 4. Unsupervised Learning Production Class
Explores advanced density estimators, hierarchical clusters, dimensionality reduction, and practical use-cases.

*   **[GMM.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Production%20Class/GMM.ipynb)**
    *   **Core Concepts:** Gaussian Mixture Models, Expectation-Maximization (EM), soft clustering, anomaly/outlier detection.
    *   **Description:** Explores probabilistic clustering with GMMs and applies density estimation to isolate statistical anomalies/outliers.
    *   **Key Libraries:** `sklearn.mixture.GaussianMixture`, `matplotlib.pyplot`.
*   **[HDBSCAN.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Production%20Class/HDBSCAN.ipynb)**
    *   **Core Concepts:** Hierarchical DBSCAN, density-based hierarchy, cluster stability.
    *   **Description:** Runs HDBSCAN on complex synthetic shapes to show how it solves the variable-density clustering problem without requiring a globally fixed epsilon.
    *   **Key Libraries:** `sklearn.cluster.HDBSCAN`, `matplotlib.pyplot`.
*   **[Image Segmentation.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Production%20Class/Image%20Segmentation.ipynb)**
    *   **Core Concepts:** Color quantization, image compression, segmenting shapes.
    *   **Description:** Practical project using K-Means to reduce the color space of sample images, effectively segmenting background and foreground objects.
    *   **Key Libraries:** `sklearn.cluster.KMeans`, `sklearn.datasets.load_sample_image`, `matplotlib.pyplot`.
*   **[PCA.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Production%20Class/PCA.ipynb)**
    *   **Core Concepts:** Dimensionality reduction, variance retention, eigenvector projections.
    *   **Description:** Implements Principal Component Analysis to compress and visualize high-dimensional datasets while explaining cumulative variance.
    *   **Key Libraries:** `sklearn.decomposition.PCA`, `matplotlib.pyplot`.
*   **[kmeans SSL.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Unsupervised%20Learning%20Production%20Class/kmeans%20SSL.ipynb)**
    *   **Core Concepts:** Semi-Supervised Learning (SSL), label propagation, clustering representative instances.
    *   **Description:** Combines K-Means clustering and Logistic Regression on digits dataset to perform label propagation, training a highly accurate model with very few labeled target variables.
    *   **Key Libraries:** `sklearn.cluster.KMeans`, `sklearn.linear_model.LogisticRegression`, `sklearn.model_selection.train_test_split`.

---

### 5. Deep Learning Foundation Class
Focuses on GPU-accelerated computing foundations, kernel programming, and custom backpropagation using OpenAI Triton.

*   **[single_perceptron_triton.py](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Deep%20Learning%20Foundation%20Class/single_perceptron_triton.py)**
    *   **Core Concepts:** Triton GPU programming, memory layout, thread indexing, loading/storing data.
    *   **Description:** Implements a single perceptron activation step as a Triton kernel, executing parallel inputs directly on the GPU.
    *   **Key Libraries:** `triton`, `triton.language`, `torch`.
*   **[multiple_perceptron_triton.py](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Deep%20Learning%20Foundation%20Class/multiple_perceptron_triton.py)**
    *   **Core Concepts:** Block matrix loading, matrix multiplication, bias tile accumulation, block masking.
    *   **Description:** Expands Triton implementations to a multi-neuron perceptron layer, performing parallelized block matrix multiplication and thresholding activation.
    *   **Key Libraries:** `triton`, `triton.language`, `torch`.
*   **[Backpropagation_triton.py](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Deep%20Learning%20Foundation%20Class/Backpropagation_triton.py)**
    *   **Core Concepts:** Custom backward pass, gradient accumulator tiles, parallel gradient reduction.
    *   **Description:** Custom Triton kernel implementing backpropagation to compute gradients for both weights ($dW$) and biases ($db$) in parallel.
    *   **Key Libraries:** `triton`, `triton.language`, `torch`.
*   **[perceptron.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Deep%20Learning%20Foundation%20Class/perceptron.ipynb)**
    *   **Core Concepts:** Autograd validation, mathematical verification, GPU execution.
    *   **Description:** Jupyter notebook to test and validate output correctness of all three custom Triton kernels against PyTorch's native autograd engine.
    *   **Key Libraries:** `torch`, `triton`.

---

### 6. Deep Learning Production Class
Deals with productionizing deep learning architectures, building input pipelines, and scaling model training.

*   **[Image Classifier.ipynb](file:///c:/Users/Ayush/Git%20Repo/Deep%20dive%20into%20AI/Deep%20Learning%20Production%20Class/Image%20Classifier.ipynb)**
    *   **Core Concepts:** Production data pipelines, dataset normalization, batch configurations, PyTorch CUDA pipeline.
    *   **Description:** Configures a production image classification pipeline using standard torchvision transforms, custom batch loaders, and datasets (MNIST) targeting GPU hardware.
    *   **Key Libraries:** `torch`, `torchvision`, `torch.utils.data.DataLoader`.

---

## 🚀 Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AyushRawat-ghost/Deep-dive-into-AI.git
    cd Deep-dive-into-AI
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate on Windows:
    .\venv\Scripts\activate
    # Activate on macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    You can install the necessary packages using:
    ```bash
    # Install core ML packages
    pip install numpy pandas matplotlib scikit-learn jupyterlab xgboost lightgbm catboost

    # Install Deep Learning dependencies (PyTorch & torchvision)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Install Triton (Note: for Windows, use triton-windows)
    pip install -U "triton-windows<3.8"
    ```

4.  **Launch JupyterLab:**
    ```bash
    jupyter lab
    ```

---

## 📝 Key Learning Outcomes
*   **Foundation vs. Production Transition:** Observe how algorithms implemented using simple scikit-learn models adapt to high-performance libraries like XGBoost, LightGBM, and CatBoost in production scenarios.
*   **Data Preparation:** Gain skills in Pipeline-based grid searches, anomaly detection with GMMs, dimensionality reduction (PCA), and scaling.
*   **Ensembling:** Understand Bagging, Boosting (Ada/Gradient), and Stacking architecture styles.
*   **Label Optimization:** Harness clustering (K-Means) to perform Semi-Supervised Learning, saving manual labeling effort.
*   **GPU Kernel Programming:** Author, compile, and run custom GPU kernels directly using OpenAI Triton language for matrix operations and forward/backward propagation.
*   **Deep Learning Pipelines:** Construct production-ready input preprocessing pipelines, dataset normalizations, and data loaders in PyTorch.
