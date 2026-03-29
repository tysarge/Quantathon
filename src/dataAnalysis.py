import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def analyze_data():
    importeddata = pd.read_csv("data/Columbus v4(FINAL TESTING ONLY).csv")
    print(importeddata.head())

    data = importeddata.drop(columns=['parcel_id', 'zip_code', 'property_type', 'last_sale_year', 'ownership_length', 'owner_occupied'])
    print(data.head())

    filtered_data = data[data["line_material"] != "copper"]
    print(filtered_data.to_string())


    writer = csv.writer(open("data/filtered_data.csv", "w"))
    writer.writerows(filtered_data.values)
 
def graph_data():
    df = pd.read_csv("data/Columbus v4(FINAL TESTING ONLY).csv")
    colors = {'galvanized': '#E07B39', 'copper': '#4A90D9', 'lead': '#C0392B'}
    markers = {'galvanized': 's', 'copper': 'o', 'lead': 'D'}
    
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#F9F9F9')
    ax.set_facecolor('#F9F9F9')
    
    for material, group in df.groupby('line_material'):
        ax.scatter(group['year_built'], group['lead_risk'],
                color=colors[material], marker=markers[material],
                s=120, zorder=5, edgecolors='white', linewidths=0.8,
                label=material.capitalize())
    
    ax.set_xlabel('Year Built', fontsize=12)
    ax.set_ylabel('Lead Risk Score', fontsize=12)
    ax.set_title('Lead Risk vs. Year Built by Pipe Material', fontsize=14, fontweight='bold', pad=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.43, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Risk Threshold (0.43)')
    ax.legend(frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('lead_risk_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
 


graph_data()