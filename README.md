# Heart Disease Prediction & Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-yellow.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red.svg)](https://matplotlib.org/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg)]()

## 📊 Project Overview

This project implements a **comprehensive heart disease prediction system** using advanced machine learning techniques and extensive data analysis. The system provides **high-accuracy medical predictions** with **detailed visualization insights**, featuring multiple ML algorithms, clustering analysis, and hyperparameter optimization for medical diagnosis support.

### 🎯 Key Objectives

✔ **High-Accuracy Predictions**: Multi-algorithm approach with 85%+ accuracy rates  
✔ **Comprehensive Analysis**: EDA with 10+ visualizations and statistical insights  
✔ **Feature Engineering**: Advanced feature selection using RF, RFE, and Chi-Square methods  
✔ **Model Optimization**: GridSearch and RandomizedSearch hyperparameter tuning  
✔ **Clustering Insights**: K-Means and Hierarchical clustering for pattern discovery  
✔ **Production Ready**: Automated model saving and prediction pipeline  
✔ **Medical Validation**: Cross-validation with medical domain expertise  
✔ **Interpretability**: Clear model explanations and feature importance analysis  

---

⚠️ **Note**: All steps are included in **one notebook**:
`notebooks/Heart_Disease_Analysis.ipynb`
Instead of splitting into multiple notebooks, each required section (01→06) is clearly labeled with headers inside this notebook:

1. **Data Preprocessing**
2. **PCA Analysis**
3. **Feature Selection** (Random Forest, RFE, Chi2)
4. **Supervised Learning** (Logistic Regression, Decision Tree, Random Forest, SVM)
5. **Unsupervised Learning** (K-Means, Hierarchical Clustering)
6. **Hyperparameter Tuning** (GridSearchCV & RandomizedSearchCV)
## 🗂️ Project Structure

```
Heart-Disease-ML-Analysis/
│
├── 📁 data/
│   ├── heart_disease.csv                 # Main dataset (303 patients)
│
├── 📁 notebooks/
│   ├── Heart_Disease_Analysis.ipynb      # Complete single notebook (01→06 sections)
│   └── exploratory_data_analysis.py
│
├── 📁 models/
│   ├── final_heart_disease_model.pkl    # Production-ready model
│   ├── scaler.pkl                        # Feature scaler
│   └── selected_features.pkl             # Top features list
│
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── Heart_Disease_Analysis_Code.md       # Complete code with headers
└── LICENSE                              # MIT License
```

## 📊 Medical Disclaimer

⚠️ **Important**: This system is designed for **research and educational purposes only**. The predictions should **never replace professional medical diagnosis** or clinical judgment. Always consult qualified healthcare professionals for medical decisions.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📞 Contact

**Ahmed Maher Abd Rabbo**
- 💼 [LinkedIn](https://www.linkedin.com/in/ahmed-maherr/)
- 📊 [Kaggle](https://kaggle.com/ahmedmaherabdrabbo)
- 📧 Email: ahmedbnmaher1@gmail.com
- 💻 [GitHub](https://github.com/AhmedMaherAbdRabbo)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.