
# Heart Disease Analysis - Complete Code

## 1. Library Imports and Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style and colors
plt.style.use('seaborn-v0_8')
PRIMARY_COLOR = '#2E86AB'
SECONDARY_COLOR = '#A23B72'
ACCENT_COLOR = '#F18F01'
SUCCESS_COLOR = '#C73E1D'

"""## 2. Data Loading and Preprocessing"""

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load the dataset
df = pd.read_csv('heart_disease.csv', names=column_names)
df = df.replace('?', np.nan)

print(f"Dataset Shape: {df.shape}")

df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)

df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)

"""## 3. Data Exploration and Information"""

df.info()

df.head()

df.describe()

missing_values = df.isnull().sum()
missing_values

df.duplicated().sum()

# Create binary target variable for classification
df['heart_disease'] = (df['target'] > 0).astype(int)

"""## 4. Data Visualization - Age Distribution"""

plt.figure(figsize=(7, 5))
plt.hist(df['age'], bins=20, color=PRIMARY_COLOR, alpha=0.7, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

"""## 5. Data Visualization - Heart Disease Distribution"""

target_counts = df['heart_disease'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(target_counts.values, labels=['No Disease', 'Disease'], colors=[SECONDARY_COLOR, ACCENT_COLOR], autopct='%1.1f%%')
plt.title('Heart Disease Distribution')
plt.show()

"""## 6. Data Visualization - Gender vs Heart Disease"""

plt.figure(figsize=(7, 5))
pd.crosstab(df['sex'], df['heart_disease']).plot(kind='bar', color=[SECONDARY_COLOR, ACCENT_COLOR])
plt.title('Gender vs Heart Disease')
plt.xlabel('Sex (0=Female, 1=Male)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)
plt.show()

"""## 7. Data Visualization - Chest Pain Type vs Heart Disease"""

plt.figure(figsize=(7, 5))
pd.crosstab(df['cp'], df['heart_disease']).plot(kind='bar', color=[SECONDARY_COLOR, ACCENT_COLOR])
plt.title('Chest Pain Type vs Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)
plt.show()

"""## 8. Correlation Analysis"""

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

"""## 9. Feature Scaling and PCA"""

X = df.drop(['target', 'heart_disease'], axis=1)
y = df['heart_disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components for 95% variance: {n_components_95}")

pca_final = PCA(n_components=n_components_95)
X_pca_final = pca_final.fit_transform(X_scaled)
print(f"PCA transformed shape: {X_pca_final.shape}")

"""## 10. Feature Selection - Random Forest Feature Importance

"""

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_scaled, y)

feature_importance = pd.DataFrame({'feature': X.columns,'importance': rf_selector.feature_importances_}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

"""## 11. Feature Selection - Recursive Feature Elimination (RFE)"""

rfe_selector = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=8)
X_rfe = rfe_selector.fit_transform(X_scaled, y)

rfe_features = X.columns[rfe_selector.support_]
print(f"Selected features by RFE: {list(rfe_features)}")

"""## 12. Feature Selection - Chi-Square Test"""

chi2_selector = SelectKBest(chi2, k=8)
X_chi2 = chi2_selector.fit_transform(X_scaled + abs(X_scaled.min()), y)

chi2_features = X.columns[chi2_selector.get_support()]
print(f"elected features by Chi2: {list(chi2_features)}")

selected_features = feature_importance.head(8)['feature'].values
X_selected = X_scaled[selected_features]
print(f"Using top {len(selected_features)} features for modeling:")
print(list(selected_features))

"""## 13. Final Feature Selection"""

chi2_selector = SelectKBest(chi2, k=8)
X_chi2 = chi2_selector.fit_transform(X_scaled + abs(X_scaled.min()), y)
chi2_features = X.columns[chi2_selector.get_support()]
print(f"Selected features by Chi2: {list(chi2_features)}")

selected_features = feature_importance.head(8)['feature'].values
X_selected = X_scaled[selected_features]
print(f"Using top {len(selected_features)} features for modeling:")
print(list(selected_features))

"""## 14. Train-Test Split"""

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2,random_state=42, stratify=y)

"""## 15. Model Training and Evaluation"""

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

results = {}
model_objects = {}

for name, model in models.items():
    print(f"Training {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    model_objects[name] = model

    print(f"   {name} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"   ROC-AUC: {roc_auc:.4f}\n")

# Results comparison
results_df = pd.DataFrame(results).T
print(f"Model Comparison:")
print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].round(4))

"""## 16. Confusion Matrix"""

cm = confusion_matrix(y_test, results['Logistic Regression']['predictions'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm = confusion_matrix(y_test, results['Decision Tree']['predictions'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm = confusion_matrix(y_test, results['Random Forest']['predictions'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm = confusion_matrix(y_test, results['SVM']['predictions'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## 17. Comparison Visualization"""

plt.figure(figsize=(7,5))
values = results_df['Accuracy'].values
model_names = results_df.index

bars = plt.bar(model_names, values, color=[PRIMARY_COLOR, SECONDARY_COLOR,
                                           ACCENT_COLOR, SUCCESS_COLOR])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.show()

plt.figure(figsize=(7,5))
values = results_df['Precision'].values
model_names = results_df.index

bars = plt.bar(model_names, values, color=[PRIMARY_COLOR, SECONDARY_COLOR,
                                           ACCENT_COLOR, SUCCESS_COLOR])
plt.title('Precision Comparison')
plt.ylabel('Precision')
plt.xticks(rotation=45)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.show()

plt.figure(figsize=(7,5))
values = results_df['Recall'].values
model_names = results_df.index

bars = plt.bar(model_names, values, color=[PRIMARY_COLOR, SECONDARY_COLOR,
                                           ACCENT_COLOR, SUCCESS_COLOR])
plt.title('Recall Comparison')
plt.ylabel('Recall')
plt.xticks(rotation=45)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.show()

plt.figure(figsize=(7,5))
values = results_df['F1-Score'].values
model_names = results_df.index

bars = plt.bar(model_names, values, color=[PRIMARY_COLOR, SECONDARY_COLOR,
                                           ACCENT_COLOR, SUCCESS_COLOR])
plt.title('F1-Score Comparison')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.show()

"""## 18. ROC Curves Comparison"""

plt.figure(figsize=(10, 8))
for name in models.keys():
    if results[name]['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        roc_auc = results[name]['ROC-AUC']
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

"""## 19. Unsupervised Learning - K-Means Clustering"""

print("UNSUPERVISED LEARNING - CLUSTERING")
print("K-Means Clustering")

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

"""## 20. K-Means Clustering Visualization

"""

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-Means Clustering Results')
plt.colorbar(scatter1, label='Cluster')

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Actual Target Labels')
plt.colorbar(scatter2, label='Target')

plt.tight_layout()
plt.show()

"""## 21. Hierarchical Clustering"""

print("Hierarchical Clustering")

linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_scaled)

"""## 22. Hierarchical Clustering Visualization"""

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter3 = plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], c=agg_labels, cmap='viridis', alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Hierarchical Clustering Results')
plt.colorbar(scatter3, label='Cluster')

plt.subplot(1, 2, 2)
scatter4 = plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Actual Target Labels')
plt.colorbar(scatter4, label='Target')

plt.tight_layout()
plt.show()

"""## 23. Clustering Results Comparison"""

print("\nK-Means Clustering vs Actual Labels:")
kmeans_comparison = pd.crosstab(kmeans_labels, y, margins=True)
print(kmeans_comparison)

print("\nHierarchical Clustering vs Actual Labels:")
agg_comparison = pd.crosstab(agg_labels, y, margins=True)
print(agg_comparison)

"""## 24. Hyperparameter Tuning - Grid Search for Random Forest"""

print("HYPERPARAMETER TUNING")
print("Grid Search for Random Forest")

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_

print("Best Random Forest Parameters:")
print(rf_grid_search.best_params_)
print(f"Best Cross-validation Score: {rf_grid_search.best_score_:.4f}")

"""## 25. Hyperparameter Tuning - Randomized Search for SVM"""

print("Randomized Search for SVM")
svm_param_dist = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm_random_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

svm_random_search.fit(X_train, y_train)
best_svm = svm_random_search.best_estimator_

print("Best SVM Parameters:")
print(svm_random_search.best_params_)
print(f"Best Cross-validation Score: {svm_random_search.best_score_:.4f}")

"""## 26. Evaluating Optimized Models"""

print("Evaluating Optimized Models")

best_rf_pred = best_rf.predict(X_test)
best_rf_proba = best_rf.predict_proba(X_test)[:, 1]

print("Optimized Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, best_rf_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, best_rf_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, best_rf_proba):.4f}")

# Evaluate optimized SVM
best_svm_pred = best_svm.predict(X_test)
best_svm_proba = best_svm.predict_proba(X_test)[:, 1]

print("\nOptimized SVM Performance:")
print(f"Accuracy: {accuracy_score(y_test, best_svm_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, best_svm_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, best_svm_proba):.4f}")

# Compare with baseline models
baseline_rf_score = results['Random Forest']['Accuracy']
baseline_svm_score = results['SVM']['Accuracy']

print(f"Random Forest - Baseline: {baseline_rf_score:.4f}, Optimized: {accuracy_score(y_test, best_rf_pred):.4f}")
print(f"SVM - Baseline: {baseline_svm_score:.4f}, Optimized: {accuracy_score(y_test, best_svm_pred):.4f}")

"""## 27. Final Model Selection and Saving"""

print("FINAL MODEL SELECTION AND SAVING")

all_models = {
    'Logistic Regression': (model_objects['Logistic Regression'], results['Logistic Regression']['Accuracy']),
    'Decision Tree': (model_objects['Decision Tree'], results['Decision Tree']['Accuracy']),
    'Random Forest': (model_objects['Random Forest'], results['Random Forest']['Accuracy']),
    'SVM': (model_objects['SVM'], results['SVM']['Accuracy']),
    'Optimized Random Forest': (best_rf, accuracy_score(y_test, best_rf_pred)),
    'Optimized SVM': (best_svm, accuracy_score(y_test, best_svm_pred))
}

best_model_name = max(all_models, key=lambda x: all_models[x][1])
best_model = all_models[best_model_name][0]
best_accuracy = all_models[best_model_name][1]

print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Build final pipeline
from sklearn.pipeline import Pipeline
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', best_model)
])
final_pipeline.fit(X_train, y_train)
print("Saving the final model...")
joblib.dump(final_pipeline, 'final_heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print("Model saved successfully!")

"""## 29. Prediction Examples"""

print("PREDICTION EXAMPLES")
sample_indices = [0, 1, 2, 3, 4]
sample_data = X_test.iloc[sample_indices]
sample_targets = y_test.iloc[sample_indices]

predictions = best_model.predict(sample_data)
probabilities = best_model.predict_proba(sample_data)

print("Sample Predictions:")
for i, idx in enumerate(sample_indices):
    actual = sample_targets.iloc[i]
    predicted = predictions[i]
    prob_no_disease = probabilities[i][0]
    prob_disease = probabilities[i][1]

    print(f"Sample {i+1}:")
    print(f"  Actual: {'Disease' if actual == 1 else 'No Disease'}")
    print(f"  Predicted: {'Disease' if predicted == 1 else 'No Disease'}")
    print(f"  Probability No Disease: {prob_no_disease:.3f}")
    print(f"  Probability Disease: {prob_disease:.3f}")
    print()

