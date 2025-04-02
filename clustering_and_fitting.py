#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:26:47 2025

@author: jacintanugwa
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def preprocessing(df):
    """Clean and prepare the heart disease data for analysis"""
    "Renaming the Column Names"
    df= df.rename(columns={
    'age':'age',
    'sex':'sex',
    'cp':'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol':'cholesterol',
    'fbs':'fasting_blood_sugar',
    'restecg':'rest_ecg',
    'thalach': 'max_heart_rate_achieved',
    'exang':'exercise_induced_angina',
    'oldpeak':'st_depression',
    'slope':'st_slope',
    'ca':'num_major_vessels',
    'thal':'thalassemia' 
    })

    print(df.columns.tolist())

    "Basic Data Exploration"
    df.describe()
    print("Data Summary:")
    print (df.describe(include='all'))
    print("\nFirst few rows:")
    print(df.head())
    print("\nbottom rows:")
    print(df.tail())

    "Replace numeric values with string label"
    df.loc[df['chest_pain_type'] == 0, 'chest_pain_type'] = 'typical angina'
    df.loc[df['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
    df.loc[df['chest_pain_type'] == 2, 'chest_pain_type'] = 'non-angina pain'
    df.loc[df['chest_pain_type'] == 3, 'chest_pain_type'] = 'asymptomatic'
    df.isna().sum()
    df = df.dropna()
    df = df.drop_duplicates()
    return df
    


def plot_relational_plot(df):
    """Creates relational plot (scatter plot)"""
    plt.figure(figsize=(12,10))
    sns.scatterplot(data=df, x='age',y='resting_blood_pressure',
    hue='target',)
    plt.title('Distribution of Blood pressure vs age', fontweight='bold')
    plt.savefig('relational_plot.png')
    plt.show()
    plt.close()
        
    "Creates relationale plot (scatter plot)"
    plt.figure(figsize=(12,10))
    sns.scatterplot(data=df, x='age',y='cholesterol', hue='target',)
    plt.title('Distribution of Cholesterol vs age', fontweight='bold')
    plt.savefig('relational_plot.png')
    plt.show()
    plt.close()




def plot_categorical_plot(df):
    """Creates a categorical plot (barplot)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='chest_pain_type', y='age',hue='target', data=df, 
    errorbar=None, palette=[ "blue","purple"])
    plt.title("A distribution of Chest pain types vs age", 
              fontsize=16,fontweight='bold')
    plt.xlabel("Chest pain Types", fontsize=12)
    plt.ylabel("Age Range", fontsize=12)
    plt.xticks(rotation=50, ha='right')
    plt.legend(title='target', bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.tight_layout()
    plt.savefig("categorical_plot.png")  
    plt.show()
    plt.close()
         
    
    plt.bar(df['chest_pain_type'],df['target'],color='red')
    plt.title("A Bar Plot of Chest pain types", fontsize=12,fontweight='bold')
    plt.show()
    plt.close()
     

def plot_statistical_plot(df):
    """Creates a statistical plot (correlation heatmap)."""
    corr_matrix = df[["cholesterol", 'resting_blood_pressure',"fasting_blood_sugar","age", ]].corr()
    fig, ax= plt.subplots(figsize=(8, 5), dpi=144)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,ax=ax)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold' )
    
    # Show the plot
    plt.show()
    

def statistical_analysis(df, col: str):
    """Calculates statistical analysis of the data"""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    kurtosis = df[col].kurtosis()
    return mean, stddev, skew, kurtosis

def writing(moments, col):
    """Interprets statistical moments in human-readable format"""
    mean, stddev, skew, kurtosis = moments
    
    print(f"\nStatistical Analysis for {col}:")
    print("="*40)
    print(f"Mean: {mean:,.2f}")
    print(f"Standard Deviation: {stddev:,.2f}")
    print(f"Skewness: {skew:.2f}")
    print(f"Kurtosis: {kurtosis:.2f}")
    
    # Interpretation
    skew_text = "right-skewed" if skew > 2 else "left-skewed" if skew < -2 else "approximately symmetric"
    kurtosis_text = "leptokurtic (peaked)" if kurtosis > 1 else "platykurtic (flat)" if kurtosis < -1 else "mesokurtic (normal)"
    
    print("\nInterpretation:")
    print(f"- The distribution is {skew_text}")
    print(f"- The distribution is {kurtosis_text} compared to normal distribution")

def perform_clustering(df, col1, col2,):
    """
    Performs clustering using KMeans and visualizes clusters.
    """
    def plot_elbow_method(wcss):
            """
            Plots the elbow method to determine the optimal number of clusters.
            """
            plt.figure(figsize=(8, 5))
            plt.plot(range(2, 11), wcss, 'kx-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            plt.title('Elbow Method', fontweight='bold')
            plt.savefig('elbow_plot.png')
            plt.show()
    
    def one_silhouette_inertia(n, xy):
            """
            Calculates the silhouette score and inertia for a given number of clusters.
            """
            kmeans = KMeans(n_clusters=n, n_init=20)
            kmeans.fit(xy)
            labels = kmeans.labels_
            score = silhouette_score(xy, labels)
            inertia = kmeans.inertia_
            return score, inertia
    
        # Scale the data
    col1= 'cholesterol'
    col2= 'resting_blood_pressure'
    data = df[[col1, col2]]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
        # Find the best number of clusters using silhouette score and inertia
    wcss = []
    best_n, best_score = None, -np.inf
    for n in range(2, 11):
            score, inertia = one_silhouette_inertia(n, data_scaled)
            wcss.append(inertia)
            if score > best_score:
                best_n = n
                best_score = score
    print(f"{n:2g} clusters silhoutte score ={ score:0.2f} ")
    print(f"Best number of clusters = {best_n}")
    plot_elbow_method(wcss)
    
    
    
    # Fit the KMeans algorithm with the best number of clusters
    kmeans = KMeans(n_clusters=best_n, n_init=20)
    kmeans.fit(data_scaled)
    df["Cluster"] = kmeans.labels_
    labels = kmeans.labels_
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = cen [:, 0]
    ykmeans = cen [:, 1]
    centre_labels = kmeans.predict(kmeans. cluster_centers_)
    return df, col1, col2, labels, data, xkmeans, ykmeans, centre_labels, best_n
    
    
def plot_clustered_data(df, col1, col2, labels, data, xkmeans, ykmeans, centre_labels, best_n):
    "Visualize clusters"
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data= df, x=col1, y=col2, hue="Cluster", palette="viridis")
    plt.title(f"K-Means Clustering ({best_n} Clusters)", fontweight='bold')
    plt.savefig('clustering.png')
    #return df["Cluster"], data_scaled, kmeans.cluster_centers_
    plt.show()
    #return df["Cluster"], data_scaled, kmeans.cluster_centers_

def perform_fitting(df, col1, col2):
    """
    Performs linear fitting (least squares).
    """
    def linear_model(x, a, b):
        """
        Defines a linear model (y = a * x + b).
        """
        return a * x + b

    x_data = df[col1]
    y_data = df[col2]
    
    # Fit the model
    params, _ = curve_fit(linear_model, x_data, y_data)
    
    # Return data for plotting
    return x_data, y_data, linear_model(x_data, *params)

def plot_fitted_data(x, y, fitted_y):
    """
    Plots the fitted data (scatter and fitted line).
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data")
    plt.plot(x, fitted_y, color='red', label="Fitted Line")
    plt.xlabel("cholesterol")
    plt.ylabel("resting_blood_pressure")
    plt.title("Linear Fit: Cholestrol vs. resting blood pressure", fontweight='bold')
    plt.legend()
    plt.savefig('fitting.png')
    plt.show()




def main():
    """Main function that executes data processing"""
    try:
    
       df = pd.read_csv('data.csv')
       df = preprocessing(df)
       
       
       plot_relational_plot(df)
       plot_statistical_plot(df)
       plot_categorical_plot(df)
        
       
       col = 'cholesterol'
       moments = statistical_analysis(df, col)
       writing(moments, col)
       print("\nAnalysis completed successfully for cholesterol!")
        
       col = 'rest_ecg'
       moments = statistical_analysis(df, col)
       writing(moments, col)
       print("\nAnalysis completed successfully for restecg!")
        
       col = 'resting_blood_pressure'
       moments = statistical_analysis(df, col)
       writing(moments, col)
       print("\nAnalysis completed successfully for resting blood pressure!")
        
       clustering_results = perform_clustering(df, 'cholesterol', 'resting_blood_pressure')
       plot_clustered_data(*clustering_results)
       fitting_results = perform_fitting(df, 'cholesterol', 'resting_blood_pressure')
       plot_fitted_data(*fitting_results)
       return


    except FileNotFoundError:
        print("Error: data.csv file not found")
    except Exception as e:
        print(f"An error occurred: {repr(e)}")

if __name__ == '__main__':
    main()
