# NGUYEN Nhat-Vy Jessica - BIA 2
# Advanced Data Visualization - PROJECT

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.manifold import Isomap
from sklearn.metrics import confusion_matrix, log_loss


# Data loading
df = pd.read_csv("sleep_health_lifestyle_dataset.csv")
df.dropna(inplace=True)

st.title("😴 Impact of Stress and Lifestyle on Sleep")
st.write("An interactive exploration of sleep data to understand how differents factors can affect our rest.")

st.sidebar.title("😴 Impact of Stress and Lifestyle on Sleep")
st.sidebar.write("[Sleep Patterns Dataset](https://www.kaggle.com/datasets/siamaktahmasbi/insights-into-sleep-patterns-and-daily-habits/data)")

st.sidebar.text("By Nhat-Vy Jessica NGUYEN")
page = st.sidebar.radio("Sleep Health Lifestyle",["📖 Visualisation and Storytelling", "📊 Manifold learning for Visualisation"])



# PART 1: Visualisation and Storytelling
if page == "📖 Visualisation and Storytelling":

    # Sidebar for navigation
    tabs = ["Introduction & Data Overview", "Data Exploration", "Factors Impact", "Conclusion"]
    page = st.radio("📖 Storytelling", tabs, key="tabs", horizontal=True)


    # Introduction & Data Overview
    if page == "Introduction & Data Overview":
        st.header("📌 Introduction: Why Study Sleep?")
        st.write("""
        Sleep is essential for our physical and mental well-being. 
        This study aims to understand the factors that influence sleep quality.
                 
        📊 We will explore:
        - 📌 General sleep trends
        - 🚀 The impact of stress and lifestyle habits
        - 🔎 Differences by age and gender
        """)

        st.header("🔍 Data Overview")
        st.write("Here’s a quick look at the first few rows of the dataset:")
        st.write(df.head())

        st.write("🔎 Data Types & Basic Information:")
        st.write(df.dtypes)

        # Missing values
        st.subheader("🧹 Data Cleaning & Preprocessing")
        st.write("Checking for missing values:")
        missing_values = df.isnull().sum()
        st.write(missing_values)

        if missing_values.any():
            st.write("Rows with missing values:")
            st.write(df[df.isnull().any(axis=1)])

        # Duplicates ?
        st.write("🚫 Checking for duplicates:")
        duplicates = df.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicates}")
        if duplicates > 0:
            st.write("Removing duplicates...")
            df = df.drop_duplicates()
            st.write(f"Data after duplicates removed: {df.shape[0]} rows")

        # Summary
        st.subheader("📊 Data Summary")
        st.write("Here’s a statistical summary of the dataset:")
        st.write(df.describe())

        st.write("🔢 Checking unique values in categorical columns:")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            st.write(f"Unique values in {col}: {df[col].nunique()}")
            st.write(df[col].unique())



    # Data exploration
    elif page == "Data Exploration":
        st.header("🔍 Data Exploration")

        # Filters
        age_range = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (20, 50))
        gender = st.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
        stress_level = st.slider("Stress Level", int(df["Stress Level (scale: 1-10)"].min()), int(df["Stress Level (scale: 1-10)"].max()), (0, 10))

        filtered_df = df[
            (df["Age"].between(age_range[0], age_range[1])) &
            (df["Gender"].isin(gender)) &
            (df["Stress Level (scale: 1-10)"].between(stress_level[0], stress_level[1]))
        ]

        st.write(f"📊 Filtered Data ({filtered_df.shape[0]})")

        # Key KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Sleep Duration", f"{filtered_df['Sleep Duration (hours)'].mean():.1f} h")
        col2.metric("Average Sleep Quality", f"{filtered_df['Quality of Sleep (scale: 1-10)'].mean():.1f}/10")
        col3.metric("Average Stress Level", f"{filtered_df['Stress Level (scale: 1-10)'].mean():.1f}/10")

        # Histogram for sleep duration
        fig = px.histogram(filtered_df, x="Sleep Duration (hours)", nbins=20, title="Sleep Duration Distribution", color="Gender")
        st.plotly_chart(fig)

        # Scatter plot between sleep duration and quality
        fig = px.scatter(
            filtered_df, x="Sleep Duration (hours)", y="Quality of Sleep (scale: 1-10)",
            color="Stress Level (scale: 1-10)", size="Sleep Duration (hours)",
            title="Link Between Sleep Duration and Quality"
        )
        st.plotly_chart(fig)

        # Bar plot (gender, quality and stress) 
        fig = px.bar(filtered_df, x="Gender", y="Quality of Sleep (scale: 1-10)", color="Stress Level (scale: 1-10)", barmode="group")
        st.plotly_chart(fig)
        
        st.write("➡️ It seems that the less stressed we are, the better we sleep, both in duration and quality.")


    # Impact 
    elif page == "Factors Impact":

        st.header("🔥 Impact of Different Factors on Sleep")

        st.subheader("🔍 Summary")

        # Key KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Sleep Duration", f"{df['Sleep Duration (hours)'].mean():.1f} h")
        col2.metric("Average Sleep Quality", f"{df['Quality of Sleep (scale: 1-10)'].mean():.1f}/10")
        col3.metric("Average Stress Level", f"{df['Stress Level (scale: 1-10)'].mean():.1f}/10")

        fig = px.histogram(df, x="Sleep Duration (hours)", nbins=20, title="Sleep Duration Distribution", color="Gender")
        st.plotly_chart(fig)

        fig = px.scatter(df, x="Sleep Duration (hours)", y="Quality of Sleep (scale: 1-10)", color="Stress Level (scale: 1-10)", size="Sleep Duration (hours)")
        st.plotly_chart(fig)

        # Normalization
        df_normalized = df.copy()
        columns_to_normalize = ["Sleep Duration (hours)", "Quality of Sleep (scale: 1-10)", "Physical Activity Level (minutes/day)", 
                                "Stress Level (scale: 1-10)", "Heart Rate (bpm)", "Daily Steps"]
        df_normalized[columns_to_normalize] = df_normalized[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        
        mean_values = df_normalized[columns_to_normalize].mean()

        # Radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=mean_values.values,  
            theta=columns_to_normalize, 
            fill='toself',
            name='Average Profile'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Radar Chart des variables clés"
        )

        st.plotly_chart(fig)

        
        # Filters
        age_range = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (20, 50))
        gender = st.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
        stress_level = st.slider("Stress Level", int(df["Stress Level (scale: 1-10)"].min()), int(df["Stress Level (scale: 1-10)"].max()), (0, 10))

        filtered_df = df[
            (df["Age"].between(age_range[0], age_range[1])) &
            (df["Gender"].isin(gender)) &
            (df["Stress Level (scale: 1-10)"].between(stress_level[0], stress_level[1]))
        ]


        st.header("🔍 Impact of Physical Activity and Habits on Sleep")

        # Impact of physical activity on sleep quality
        fig = px.scatter(filtered_df, 
                        x="Physical Activity Level (minutes/day)", 
                        y="Quality of Sleep (scale: 1-10)", 
                        color="Stress Level (scale: 1-10)", 
                        title="Relationship Between Physical Activity and Sleep Quality")
        st.plotly_chart(fig)

        # Violin plot with Plotly
        fig = px.violin(df, x="Gender", y="Physical Activity Level (minutes/day)", 
                        title="Physical Activity Level by Gender", 
                        labels={'Gender': 'Gender', 'Physical Activity Level (minutes/day)': 'Physical Activity Level (minutes/day)'})
        st.plotly_chart(fig)

        st.write("""🧐 **What do we observe?** 
        Higher physical activity seems to improve sleep quality and reduce stress.
        """)

        # BMI & Heart Rate
        st.header("💪 Impact of BMI and Heart Rate")

        # Bar plot of BMI category and its impact on sleep
        fig = px.bar(filtered_df, 
                    x="BMI Category", 
                    y="Quality of Sleep (scale: 1-10)", 
                    color="Stress Level (scale: 1-10)", 
                    title="Impact of BMI on Sleep Quality")
        st.plotly_chart(fig)

        # Scatter plot: BMI vs Sleep Quality
        fig = px.scatter(filtered_df, 
                        x="BMI Category", 
                        y="Quality of Sleep (scale: 1-10)", 
                        color="Heart Rate (bpm)", 
                        title="Relationship Between BMI and Sleep Quality")
        st.plotly_chart(fig)

        st.write("""🔬 **Analysis:**
        People in extreme BMI categories (too low or too high - particularly obese and overweight ) may experience reduced sleep quality.
        People with a higher heart rate also seems to have poorer sleep quality.
        """)

        # Occupation
        st.header("💪 Impact of Stress and Occupation")

        # Boxplot with Plotly
        fig = px.box(df, x="Stress Level (scale: 1-10)", y="Quality of Sleep (scale: 1-10)", 
                    title="Sleep Quality vs. Stress Level", 
                    labels={'Stress Level (scale: 1-10)': 'Stress Level', 'Quality of Sleep (scale: 1-10)': 'Quality of Sleep'})
        st.plotly_chart(fig)

        fig = px.bar(df, x="Gender", y="Quality of Sleep (scale: 1-10)", color="Stress Level (scale: 1-10)", barmode="group")
        st.plotly_chart(fig)

        # Occupation vs Sleep Duration
        fig = px.box(filtered_df, 
                    x="Occupation", 
                    y="Sleep Duration (hours)", 
                    title="Sleep Duration Based on Occupation")
        st.plotly_chart(fig)

        st.write("""🧐 **Observations:**
        Students and manual labor individuals tend to have more irregular sleep patterns.
        """)

        st.write("📊 We can see a relationship between sleep duration and stress level.")


    # Conclusion
    elif page == "Conclusion":

        st.header("🎯 Conclusions & Recommendations")
        
        st.subheader("💡 What We Learned")
        st.write("""
        - Stress significantly impacts both sleep duration and quality
        - Young adults have more irregular sleep habits.
        - There are small differences between men and women.
        - Lifestyle factors like physical activity and BMI also play crucial roles.
        - Interventions to reduce stress could improve sleep quality and overall health.
        """)

        st.subheader("📌 Tips for Better Sleep")
        st.write("""
        - 🧘 Reduce stress with relaxation exercises
        - 📵 Avoid screens before bed
        - 🏋️ Exercise to regulate energy
        - ☕ Limit caffeine and alcohol in the evening
        """)


# PART 2: Manifold learning for Visualisation
elif page == "📊 Manifold learning for Visualisation":

    st.subheader("Sleep Data Analysis")
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Data Preprocessing: Splitting 'Blood Pressure (systolic/diastolic)'
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)
    df.drop(columns=['Blood Pressure (systolic/diastolic)'], inplace=True)
    numerical_features = ['Sleep Duration (hours)', 'Stress Level (scale: 1-10)', 'Daily Steps', 'Systolic BP', 'Diastolic BP']

    # Plot distributions
    fig, axes = plt.subplots(1, len(numerical_features), figsize=(20, 5))
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribution of {feature}")
    st.pyplot(fig)

    # Feature selection for dimensionality reduction
    features = ['Age', 'Sleep Duration (hours)', 'Quality of Sleep (scale: 1-10)', 'Physical Activity Level (minutes/day)', 'Stress Level (scale: 1-10)', 'Heart Rate (bpm)', 'Daily Steps', 'Systolic BP', 'Diastolic BP']
    X = df[features]

    # Standardizing data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(X_scaled, columns=features).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Dimensionality Reduction 2D
    st.subheader("Dimensionality Reduction")
    y = df['Sleep Disorder'].apply(lambda x: 0 if x == 'None' else (1 if x == 'Insomnia' else 2))  
    methods = {
        "PCA": PCA(n_components=2).fit_transform(X_scaled),
        "t-SNE": TSNE(n_components=2).fit_transform(X_scaled),
        "LLE": LocallyLinearEmbedding(n_components=2).fit_transform(X_scaled),
        "Isomap": Isomap(n_components=2).fit_transform(X_scaled)
    }

    fig, axes = plt.subplots(1, len(methods), figsize=(24, 6))
    for i, (name, data) in enumerate(methods.items()):
        scatter = axes[i].scatter(data[:, 0], data[:, 1], c=y, cmap='coolwarm', alpha=0.7)
        axes[i].set_title(name)
        axes[i].set_xlabel(f"{name} Component 1")
        axes[i].set_ylabel(f"{name} Component 2")

    fig.colorbar(scatter) 
    st.pyplot(fig)

    # Dimensionality Reduction 3D
    st.subheader("Dimensionality Reduction (3D)")
    y = df['Sleep Disorder'].apply(lambda x: 0 if x == 'None' else (1 if x == 'Insomnia' else 2))

    methods_3d = {
        "PCA (3D)": PCA(n_components=3).fit_transform(X_scaled),
        "t-SNE (3D)": TSNE(n_components=3).fit_transform(X_scaled),
        "LLE (3D)": LocallyLinearEmbedding(n_components=3).fit_transform(X_scaled),
        "Isomap (3D)": Isomap(n_components=3).fit_transform(X_scaled)
    }

    fig = plt.figure(figsize=(18, 6))
    for i, (name, data) in enumerate(methods_3d.items()):
        ax = fig.add_subplot(1, len(methods_3d), i + 1, projection='3d')
        sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=y, cmap='coolwarm', alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

    fig.colorbar(sc)  
    st.pyplot(fig)


    # Classification: Predicting Sleep Disorders for each reduction method
    st.subheader("Predicting Sleep Disorders (with Dimensionality Reduction)")
    for name, reduced_data in methods.items():
        st.write(f"### {name} - Model Performance")
        X_train, X_test, y_train, y_test = train_test_split(reduced_data, y, test_size=0.3, random_state=42)

        # Hyperparameter tuning for Random Forest
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        st.write("Best Parameters:", grid_search.best_params_)
        st.write("Optimized Model Score:", best_model.score(X_test, y_test))

        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.text("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Log Loss")
        st.write(f"Log Loss: {log_loss(y_test, y_prob):.4f}")

    # Learning curve for each method
    def plot_learning_curve(estimator, title, X, y, ax=None, cv=3):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
        
        if ax is None:
            ax = plt.gca()  
        
        ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score", color="blue")
        ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-validation score", color="green")
        ax.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), color="blue", alpha=0.2)
        ax.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                        np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), color="green", alpha=0.2)
        
        ax.set_title(title)
        ax.set_xlabel("Training Size")
        ax.set_ylabel("Score")
        ax.legend(loc="best")
        ax.grid()

    # Subplots
    def plot_learning_curves_for_methods(methods, methods_3d, best_model, y):
        n_2d = len(methods)
        n_3d = len(methods_3d)

        fig, axes = plt.subplots(nrows=2, ncols=max(n_2d, n_3d), figsize=(15, 10))

        for i, (name, data) in enumerate(methods.items()):
            X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
            plot_learning_curve(best_model, f"Learning Curve for {name}", X_train_reduced, y_train, ax=axes[0, i])

        for i, (name, data) in enumerate(methods_3d.items()):
            X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
            plot_learning_curve(best_model, f"Learning Curve for {name}", X_train_reduced, y_train, ax=axes[1, i])

        plt.tight_layout()
        st.pyplot(fig)

    plot_learning_curves_for_methods(methods, methods_3d, best_model, y)


    # Model Explanation and Proposed Improvements
    st.subheader("🎯 Model Explanation and Analysis")
    st.write("""
        The Random Forest model predicts sleep disorders based on features such as sleep habits, stress level, and physical activity. 
        The model's performance is evaluated using classification metrics.
    """)

    st.write("""
        How to analyse the model performance?

        1. Classification Report:
        Precision: Measures the accuracy of positive predictions.
        Recall: Measures the ability to identify all positive cases.
        F1-Score: Balances precision and recall (the higher, the better).
        Support: Indicates the number of instances per class.
        => Higher F1 scores across classes indicate a better model.

        2. Log Loss:
        Measures the quality of probabilistic predictions. 
        => Lower log loss means better model performance.

        3. Confusion Matrix:
        Shows the true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).
        => Fewer FP and FN mean the model is more accurate.
             
        Summary:
        Based on this, the best one should be ISOMAP, which have the best model score (0.72), the highest F1 score and the confusion matrix with the fewer errors.
        """)        

    st.subheader("💡 Possible Improvements")
    st.write("""
        - **Alternative Models**: Testing other classifiers such as SVM or XGBoost.
        - **Hyperparameter Optimization**: Further tuning to improve accuracy.
        - Explore the possibility of adding more sleep data to enrich the dataset and improve the quality of the analysis.
    """)
