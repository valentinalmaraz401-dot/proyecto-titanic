import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_y_explorar(filepath):
    print("=" * 60)
    print("1. EXPLORACIÓN INICIAL DEL DATASET")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"Número de pasajeros (filas): {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    print("\nVariables disponibles y tipos de datos:")
    print(df.dtypes)
    
    print("\nValores faltantes por columna:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print(f"\nRegistros duplicados: {df.duplicated().sum()}")
    
    print("\nEstadísticas descriptivas generales:")
    print(df.describe())
    
    return df

def limpieza_y_preprocesamiento(df):
    print("\n" + "=" * 60)
    print("2. LIMPIEZA Y PREPROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    df_clean = df.copy()
    
    mediana_edad = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    df_clean['Age'] = df_clean['Age'].fillna(mediana_edad)
    print("- 'Age': Valores faltantes imputados con la mediana basada en Clase y Género.")
    
    moda_embarked = df_clean['Embarked'].mode()[0]
    df_clean['Embarked'] = df_clean['Embarked'].fillna(moda_embarked)
    print(f"- 'Embarked': Valores faltantes imputados con la moda ('{moda_embarked}').")
    
    df_clean['Has_Cabin'] = df_clean['Cabin'].notnull().astype(int)
    df_clean['Cabin'] = df_clean['Cabin'].fillna('Sin Cabina')
    print("- 'Cabin': Faltantes marcados como 'Sin Cabina' y creada variable indicadora 'Has_Cabin'.")
    
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    print("- Nueva variable creada: 'FamilySize' (SibSp + Parch + 1)")
    
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
    print("- Nueva variable creada: 'IsAlone' (1 si FamilySize == 1, 0 en caso contrario)")
    
    bins = [0, 12, 24, 59, 120]
    labels = ['Niño', 'Joven', 'Adulto', 'Adulto Mayor']
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=bins, labels=labels, right=True)
    print("- Nueva variable creada: 'AgeGroup' (Niño [0-12], Joven [13-24], Adulto [25-59], Adulto Mayor [60+])")
    
    return df_clean

def realizar_analisis(df):
    print("\n" + "=" * 60)
    print("3. ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 60)
    
    tasa_supervivencia = df['Survived'].mean() * 100
    print(f"Análisis 1: Tasa general de supervivencia: {tasa_supervivencia:.2f}%")
    
    sup_sexo = df.groupby('Sex')['Survived'].mean() * 100
    print("\nAnálisis 2: Tasa de supervivencia por género:")
    print(sup_sexo.round(2).to_string())
    
    sup_clase = df.groupby('Pclass')['Survived'].mean() * 100
    print("\nAnálisis 3: Tasa de supervivencia por Clase:")
    print(sup_clase.round(2).to_string())
    
    sup_edad = df.groupby('AgeGroup', observed=False)['Survived'].mean() * 100
    print("\nAnálisis 4: Tasa de supervivencia por Grupo de Edad:")
    print(sup_edad.round(2).to_string())
    
    sup_solo = df.groupby('IsAlone')['Survived'].mean() * 100
    print("\nAnálisis 5: Tasa de supervivencia según condición de viaje (0 = Acompañado, 1 = Solo):")
    print(sup_solo.round(2).to_string())

def generar_visualizaciones(df, output_dir):
    print("\n" + "=" * 60)
    print("4. GENERACIÓN DE VISUALIZACIONES")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 5))
    ax1 = sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', errorbar=None, palette='Set2')
    plt.title('Tasa de Supervivencia por Clase y Género')
    plt.xlabel('Clase del Pasajero')
    plt.ylabel('Tasa de Supervivencia')
    plt.ylim(0, 1)
    for p in ax1.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax1.annotate(f'{height*100:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', color='white', fontweight='bold')
    path1 = os.path.join(output_dir, 'supervivencia_genero_clase.png')
    plt.savefig(path1, bbox_inches='tight')
    plt.close()
    print(f"- Gráfico guardado: {path1}")
    
    plt.figure(figsize=(8, 5))
    ax2 = sns.barplot(data=df, x='AgeGroup', y='Survived', errorbar=None, palette='Blues_d')
    plt.title('Tasa de Supervivencia por Grupo de Edad')
    plt.xlabel('Grupo de Edad')
    plt.ylabel('Tasa de Supervivencia')
    plt.ylim(0, 1)
    for p in ax2.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax2.annotate(f'{height*100:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', color='white', fontweight='bold')
    path2 = os.path.join(output_dir, 'supervivencia_grupo_edad.png')
    plt.savefig(path2, bbox_inches='tight')
    plt.close()
    print(f"- Gráfico guardado: {path2}")
    
    plt.figure(figsize=(8, 5))
    ax3 = sns.barplot(data=df, x='IsAlone', y='Survived', errorbar=None, palette='Set1')
    plt.title('Tasa de Supervivencia: Solo vs Acompañado')
    plt.xlabel('Condición (0 = Acompañado, 1 = Solo)')
    plt.ylabel('Tasa de Supervivencia')
    plt.ylim(0, 1)
    for p in ax3.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax3.annotate(f'{height*100:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', color='white', fontweight='bold')
    path3 = os.path.join(output_dir, 'supervivencia_acompanamiento.png')
    plt.savefig(path3, bbox_inches='tight')
    plt.close()
    print(f"- Gráfico guardado: {path3}")

def mostrar_conclusiones():
    conclusiones = """
============================================================
5. CONCLUSIONES Y PRINCIPALES HALLAZGOS
============================================================
1. Género: El factor determinante en la supervivencia fue el género. 
   Las mujeres registraron una tasa de supervivencia cercana al 74%, mientras 
   que los hombres apenas alcanzaron un 19%, confirmando la prioridad 'mujeres y niños primero'.

2. Clase Social: La tasa de supervivencia fue considerablemente superior en la Primera Clase (63%),
   frente a la Segunda Clase (47%) y la Tercera Clase (24%).

3. Edad: Los niños (0-12 años) tuvieron una mayor tasa de supervivencia frente a adultos y adultos mayores.

4. Acompañamiento: Viajar acompañado incrementó las probabilidades de sobrevivir (50.6%) 
   en comparación con viajar solo (30.3%).
============================================================
"""
    print(conclusiones)

if __name__ == '__main__':
    data_path = os.path.join('data', 'train.csv')
    output_directory = os.path.join('outputs', 'resultados')
    
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo {data_path}. Asegúrate de poner 'train.csv' en la carpeta 'data/'.")
    else:
        df_raw = cargar_y_explorar(data_path)
        df_clean = limpieza_y_preprocesamiento(df_raw)
        realizar_analisis(df_clean)
        generar_visualizaciones(df_clean, output_directory)
        mostrar_conclusiones()
