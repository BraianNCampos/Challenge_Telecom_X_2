import pandas as pd
import requests

# URL del archivo JSON
url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json"

# Descargar los datos
response = requests.get(url)

# Verificar si la descarga fue exitosa
if response.status_code == 200:
    data_json = response.json()

    # Convertir a DataFrame
    df = pd.DataFrame(data_json)
else:
    print("Error al obtener los datos:", response.status_code)
    exit()

# Mostrar primeras filas
print("Primeras filas del DataFrame original:")
print(df.head())

# Info general
print("\nInformación del DataFrame:")
print(df.info())

# Valores nulos por columna
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Porcentaje de nulos
print("\nPorcentaje de valores nulos por columna:")
print((df.isnull().sum() / len(df)) * 100)

# Convertir columnas tipo dict a string para detectar duplicados
cols_dict = ['customer', 'phone', 'internet', 'account']
df[cols_dict] = df[cols_dict].applymap(str)

# Filas duplicadas
print("\nFilas duplicadas:", df.duplicated().sum())

# Eliminar duplicados si hay
df = df.drop_duplicates()

# Revisar tipos
print("\nTipos de datos:")
print(df.dtypes)

# Si hay una columna 'TotalCharges', convertir a numérico
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Ver categorías únicas en columnas categóricas
print("\nCategorías únicas por columna categórica:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")

# Resumen general
print("\nResumen estadístico:")
print(df.describe(include='all'))

# Volver a convertir las columnas dict a dict (porque antes las pasamos a string)
for col in cols_dict:
    df[col] = df[col].apply(eval)

# Normalizar las columnas anidadas: 'customer', 'phone', 'internet', 'account'
cols_to_normalize = ['customer', 'phone', 'internet', 'account']
base_df = df.drop(columns=cols_to_normalize)
normalized_parts = [base_df]

for col in cols_to_normalize:
    if col in df.columns:
        norm_df = pd.json_normalize(df[col])
        normalized_parts.append(norm_df)

# Concatenar en un nuevo DataFrame
df_clean = pd.concat(normalized_parts, axis=1)

# Revisar duplicados en el nuevo DataFrame
print("\nFilas duplicadas luego de normalizar:", df_clean.duplicated().sum())

# Eliminar columnas irrelevantes
# Paso 1: Detectar columnas con un solo valor
cols_unicas = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]

# Paso 2: Identificar columnas tipo ID (por nombre)
cols_id = [col for col in df_clean.columns if 'id' in col.lower() or 'customer' in col.lower() or 'accountnumber' in col.lower()]

# Paso 3: Revisar columnas tipo fecha que no aportan
cols_fecha = [col for col in df_clean.columns if 'fecha' in col.lower() or 'date' in col.lower()]

# Combinar todas las columnas a eliminar
cols_a_eliminar = list(set(cols_unicas + cols_id + cols_fecha))

print(f"\nColumnas a eliminar por baja relevancia o ser identificadores: {cols_a_eliminar}")

# Eliminar del DataFrame
df_clean = df_clean.drop(columns=cols_a_eliminar)

# Identificar columnas categóricas (tipo 'object' o 'category')
cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nColumnas categóricas a codificar: {cat_cols}")

# Aplicar One-Hot Encoding (evitamos la trampa de multicolinealidad con drop_first=True)
df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
# Crear la variable objetivo 'Churn' y eliminar la dummy si es necesario
if 'Churn_Yes' in df_encoded.columns:
    df_encoded['Churn'] = df_encoded['Churn_Yes']
    df_encoded.drop(columns=['Churn_Yes'], inplace=True)
elif 'Churn' in df_clean.columns:
    df_encoded['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})
else:
    raise ValueError("No se encontró una columna válida de 'Churn'")

print("\nShape original:", df_clean.shape)
print("Shape luego del One-Hot Encoding:", df_encoded.shape)

# === Análisis de desbalance de clases ===

# Si 'Churn' aún está en df_clean, usarla directamente
if 'Churn' in df_clean.columns:
    target = df_clean['Churn']
else:
    # Si se codificó, buscar la columna de Churn en df_encoded
    churn_cols = [col for col in df_encoded.columns if 'Churn' in col]
    if churn_cols:
        target = df_encoded[churn_cols[0]]
    else:
        raise ValueError("No se encontró la variable 'Churn' en el DataFrame.")

# Convertir a binario si es necesario
if target.dtype == 'object':
    target = target.map({'No': 0, 'Yes': 1})

# Calcular proporciones
churn_counts = target.value_counts()
churn_percent = churn_counts / churn_counts.sum() * 100

print("\nProporción de clases en 'Churn':")
for label, count, percent in zip(churn_counts.index, churn_counts, churn_percent):
    print(f"Clase {label}: {count} clientes ({percent:.2f}%)")

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='pastel')
plt.title('Distribución de clases en la variable Churn')
plt.xlabel('Churn (0 = No, 1 = Sí)')
plt.ylabel('Número de clientes')
plt.show()

# === Estandarización de variables numéricas (para modelos sensibles a escala) ===
from sklearn.preprocessing import StandardScaler

# Seleccionar columnas numéricas del DataFrame codificado
num_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remover 'Churn' si está presente
if 'Churn' in num_cols:
    num_cols.remove('Churn')

df_encoded_no_scaling = df_encoded.copy()


# Inicializar y aplicar el escalador
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

print("\nSe aplicó estandarización a las siguientes columnas numéricas:")
print(num_cols)

# Asegurarse de que 'Churn' esté como columna binaria explícita
if 'Churn_Yes' in df_encoded.columns:
    df_encoded['Churn'] = df_encoded['Churn_Yes']
elif 'Churn' in df_clean.columns:
    # Si no se codificó como Churn_Yes, mapearla directamente desde df_clean
    df_encoded['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})
else:
    raise ValueError("No se encuentra ninguna columna válida de 'Churn' en df_encoded o df_clean")

# Confirmar que existe y es binaria
print("\nVerificación de la columna 'Churn':")
print(df_encoded['Churn'].value_counts())



# Verificar si existe una columna de fecha
fecha_column = [col for col in df_clean.columns if 'fecha' in col.lower()]
if fecha_column:
    fecha_col_name = fecha_column[0]
    df_clean['Fecha'] = pd.to_datetime(df_clean[fecha_col_name], errors='coerce')
    
    df_clean['Año'] = df_clean['Fecha'].dt.year
    df_clean['Mes'] = df_clean['Fecha'].dt.month
    df_clean['Dias_Mes'] = df_clean['Fecha'].dt.days_in_month

    # Si 'Facturacion_Mensual' existe, calcular Cuentas_Diarias
    if 'Facturacion_Mensual' in df_clean.columns:
        df_clean['Cuentas_Diarias'] = df_clean['Facturacion_Mensual'] / df_clean['Dias_Mes']
else:
    print("\nNo se encontró ninguna columna de fecha en los datos.")

# Vista final
print("\nPrimeras filas del DataFrame limpio:")
print(df_clean.head())


import pandas as pd

# Asumiendo que ya tienes el DataFrame limpio llamado df_clean

# Seleccionamos las columnas numéricas
num_cols = df_clean.select_dtypes(include=['number']).columns

print("Análisis descriptivo de columnas numéricas:\n")

desc_num = df_clean[num_cols].describe().T

# Agregar mediana (50%) para más claridad
desc_num['mediana'] = df_clean[num_cols].median()

# Mostrar el resultado con algunas métricas importantes
print(desc_num[['count', 'mean', 'mediana', 'std', 'min', '25%', '50%', '75%', 'max']])

# === Matriz de correlación ===
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurarse de que 'Churn' sea numérica
if df_clean['Churn'].dtype == 'object':
    df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})

# Seleccionar variables numéricas para correlación
corr_matrix = df_clean.select_dtypes(include='number').corr()

# Visualización con mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Matriz de correlación entre variables numéricas')
plt.tight_layout()
plt.show()

# Mostrar las correlaciones con la variable 'Churn', ordenadas
print("\nCorrelación de variables con 'Churn':")
print(corr_matrix['Churn'].sort_values(ascending=False))


# Análisis de variables categóricas
cat_cols = df_clean.select_dtypes(include=['object']).columns

print("\nAnálisis descriptivo de columnas categóricas:\n")

for col in cat_cols:
    print(f"Columna: {col}")
    conteo = df_clean[col].value_counts(dropna=False)
    porcentaje = df_clean[col].value_counts(normalize=True, dropna=False) * 100
    resumen_cat = pd.DataFrame({'Conteo': conteo, 'Porcentaje (%)': porcentaje})
    print(resumen_cat)
    print("-" * 40)


import matplotlib.pyplot as plt
import seaborn as sns

# Asumiendo que la variable 'Churn' está en df_clean o en df original si no la normalizaste
# Si está en df_clean, usar df_clean['Churn'], sino df['Churn']

# Conteo de la variable Churn
churn_counts = df_clean['Churn'].value_counts()

# Gráfico de barras
plt.figure(figsize=(6,4))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='pastel')
plt.title('Distribución de la variable Churn')
plt.xlabel('Churn')
plt.ylabel('Número de clientes')
plt.show()

# Gráfico de pastel
plt.figure(figsize=(6,6))
plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=140)
plt.title('Proporción de clientes según Churn')
plt.axis('equal')  # Para que el pie sea un círculo
plt.show()


# Variables categóricas que queremos analizar respecto a Churn
cat_vars = ['gender', 'Contract', 'PaymentMethod', 'PhoneService', 'InternetService']

for var in cat_vars:
    if var in df_clean.columns:
        plt.figure(figsize=(8, 4))
        # Conteo agrupado por la variable y el Churn
        sns.countplot(data=df_clean, x=var, hue='Churn', palette='Set2')
        
        plt.title(f'Distribución de Churn según {var}')
        plt.xlabel(var)
        plt.ylabel('Número de clientes')
        plt.legend(title='Churn')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        
        # Variables numéricas que suelen ser relevantes
# Variables numéricas que suelen ser relevantes
num_vars = ['MonthlyCharges', 'TotalCharges', 'tenure']

for var in num_vars:
    if var in df_encoded.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_encoded, x='Churn', y=var, hue='Churn', palette='pastel', legend=False)
        plt.title(f'Distribución de {var} según Churn')
        plt.xlabel('Churn')
        plt.ylabel(var)
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df_encoded, x=var, hue='Churn', fill=True, common_norm=False, alpha=0.5, palette='Set1')
        plt.title(f'Densidad de {var} para clientes que sí/no hicieron churn')
        plt.xlabel(var)
        plt.ylabel('Densidad')
        plt.show()

# === Relación entre variables específicas y cancelación ===
if 'tenure' in df_encoded.columns and 'TotalCharges' in df_encoded.columns:
    # Boxplot: Tiempo de contrato vs Cancelación
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_encoded, x='Churn', y='tenure', hue='Churn', palette='pastel', legend=False)
    plt.title('Tiempo de contrato según Cancelación')
    plt.xlabel('Churn (0 = No, 1 = Sí)')
    plt.ylabel('Meses de permanencia')
    plt.show()

    # Boxplot: Gasto total vs Cancelación
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_encoded, x='Churn', y='TotalCharges', hue='Churn', palette='pastel', legend=False)
    plt.title('Gasto total según Cancelación')
    plt.xlabel('Churn (0 = No, 1 = Sí)')
    plt.ylabel('Total Charges')
    plt.show()

    # Scatterplot: Tiempo de contrato vs Gasto total (coloreado por cancelación)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_encoded, x='tenure', y='TotalCharges', hue='Churn', palette='Set1', alpha=0.6)
    plt.title('Relación entre Tiempo de contrato y Gasto total (según Cancelación)')
    plt.xlabel('Meses de permanencia')
    plt.ylabel('Total Charges')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.show()

# === División del dataset en entrenamiento y prueba ===
from sklearn.model_selection import train_test_split

# Eliminar filas donde 'Churn' tenga NaN
df_encoded = df_encoded.dropna(subset=['Churn'])

# Asegurar que 'Churn' es la variable objetivo
if 'Churn' in df_encoded.columns:
    X = df_encoded.drop(columns='Churn')
    y = df_encoded['Churn']
else:
    raise ValueError("La variable 'Churn' no se encuentra en df_encoded")

# Eliminar columnas con 'churn' por si quedó alguna más
X = X.drop(columns=[col for col in X.columns if 'churn' in col.lower()], errors='ignore')

# Verificar que no haya columnas de fuga
print("Columnas con 'churn' en X:", [col for col in X.columns if 'churn' in col.lower()])

# Copia del dataset sin escalar, con la variable Churn incluida
df_encoded_no_scaling = df_encoded.copy()

# División 70% entrenamiento, 30% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tamaño del set de entrenamiento: {X_train.shape}")
print(f"Tamaño del set de prueba: {X_test.shape}")



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === Modelo 1: Regresión Logística (requiere normalización, usamos df_encoded ya escalado) ===

# Entrenar modelo
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Predicción
y_pred_log = log_model.predict(X_test)

# Evaluación
print("\n===== Regresión Logística =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_log))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_log))


# === Modelo 2: Random Forest (no necesita normalización) ===

# Usamos la copia sin escalar
X_rf = df_encoded_no_scaling.drop(columns='Churn')
y_rf = df_encoded_no_scaling['Churn'].astype(int)

# División 70/30
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf
)

# Entrenamiento del modelo
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Predicciones
y_pred_rf = rf_model.predict(X_test_rf)

# Evaluación
print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test_rf, y_pred_rf))
print("Matriz de Confusión:\n", confusion_matrix(y_test_rf, y_pred_rf))
print("Reporte de Clasificación:\n", classification_report(y_test_rf, y_pred_rf))


import numpy as np
coef_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Coeficiente': log_model.coef_[0]
})
coef_df['Importancia_abs'] = np.abs(coef_df['Coeficiente'])
coef_df = coef_df.sort_values(by='Importancia_abs', ascending=False)
print(coef_df.head(10))

importances = rf_model.feature_importances_
importances_df = pd.DataFrame({
    'Variable': X_rf.columns,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)
print(importances_df.head(10))
