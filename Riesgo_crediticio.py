import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = "BD-2007-2018.csv" 
    path = os.path.join(directorio_actual, nombre_archivo)

    cols_to_use = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
        'emp_length', 'home_ownership', 'annual_inc', 
        'verification_status', 'loan_status', 'dti'
    ]

    try:
        df = pd.read_csv(path, usecols=cols_to_use, low_memory=False)
        print(f"Carga exitosa desde: {path}")

        mapeo_target = {'Fully Paid': 0, 'Charged Off': 1, 'Default': 1}
        df = df[df['loan_status'].isin(mapeo_target.keys())].copy()
        df['loan_status'] = df['loan_status'].map(mapeo_target)

        df['term'] = df['term'].str.replace(' months', '', regex=False).astype(int)
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').fillna(0).astype(int)

        df['dti'] = df['dti'].fillna(df['dti'].median())
        df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())

        conn = sqlite3.connect(os.path.join(directorio_actual, "riesgo_crediticio.db"))
        df.to_sql('prestamos_limpios', conn, if_exists='replace', index=False)

        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df['grade'] = df['grade'].map(grade_map)

        df = pd.get_dummies(df, columns=['home_ownership', 'verification_status'], drop_first=True)
        df = df.dropna()

        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("Entrenando modelo mejorado... (esto puede tardar)")
        modelo_mejorado = RandomForestClassifier(
            n_estimators=100, 
            max_depth=12,  
            class_weight={0: 1, 1: 2.5}, 
            random_state=42, 
            n_jobs=-1
        )
        modelo_mejorado.fit(X_train, y_train)
        
        joblib.dump(modelo_mejorado, os.path.join(directorio_actual, 'modelo_riesgo.pkl'))
        joblib.dump(X.columns.tolist(), os.path.join(directorio_actual, 'columnas_modelo.pkl'))
        print("Modelo entrenado y guardado.")

        probabilidades = modelo_mejorado.predict_proba(X_test)[:, 1]
        umbral_optimo = 0.60
        y_pred_ajustado = (probabilidades > umbral_optimo).astype(int)

        print(f"\n REPORTE DE CLASIFICACIÓN (Umbral de decisión: {umbral_optimo*100}%):")
        print(classification_report(y_test, y_pred_ajustado))

        cm = confusion_matrix(y_test, y_pred_ajustado)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predijo Pagó', 'Predijo Default'],
                    yticklabels=['Real Pagó', 'Real Default'])
        plt.title(f'Matriz de Confusión (Umbral {umbral_optimo})')
        plt.show()

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en {path}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()