import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Citirea datelor din fișierul CSV
df = pd.read_csv("date_tema_1_iaut_2024.csv", decimal=',')

# Conversia coloanelor numerice în format float
numeric_columns = ['Regular_fiber_diet', 'Sedentary_hours_daily', 'Age', 'Est_avg_calorie_intake',
                   'Main_meals_daily', 'Height', 'Water_daily', 'Weight', 'Physical_activity_level',
                   'Technology_time_use']

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Analiza echilibrului de clase
plt.figure(figsize=(8, 6))
sns.countplot(x='Diagnostic', data=df)
plt.title('Frecvența de apariție a etichetelor')
plt.xlabel('Diagnostic')
plt.ylabel('Număr de apariții')
plt.show()

# Vizualizarea datelor numerice
numeric_attributes = df.select_dtypes(include=['float64', 'int64'])
numeric_stats = numeric_attributes.describe()
print("Statistici pentru atributele numerice:\n", numeric_stats.to_string())

# Analize de covarianță
correlation_matrix = numeric_attributes.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corelații')
plt.show()

# Vizualizarea datelor categorice
categorical_attributes = df.select_dtypes(exclude=['float64', 'int64'])
categorical_attributes = categorical_attributes.drop(columns=['Diagnostic'])
categorical_stats = categorical_attributes.describe(include=['object'])
print("Statistici pentru atributele categorice:\n", categorical_stats.to_string())

# Histograma pentru 'Smoker', 'Alcohol' și 'Snacks'
plt.figure(figsize=(18, 6))
for column in ['Smoker', 'Alcohol', 'Snacks']:
    plt.subplot(1, 3, ['Smoker', 'Alcohol', 'Snacks'].index(column) + 1)
    sns.countplot(data=categorical_attributes[column])
    plt.title(f'Histograma pentru {column}')
    plt.xlabel('Valoare')
    plt.ylabel('Contor')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histograma pentru 'Diagnostic_in_family_history', 'High_calorie_diet' și 'Calorie_monitoring'
plt.figure(figsize=(18, 6))
for column in ['Diagnostic_in_family_history', 'High_calorie_diet', 'Calorie_monitoring']:
    plt.subplot(1, 3, ['Diagnostic_in_family_history', 'High_calorie_diet', 'Calorie_monitoring'].index(column) + 1)
    sns.countplot(data=categorical_attributes[column])
    plt.title(f'Histograma pentru {column}')
    plt.xlabel('Valoare')
    plt.ylabel('Contor')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histograma pentru restul coloanelor categorice
other_categorical_attributes = categorical_attributes.drop(columns=['Smoker', 'Alcohol', 'Snacks', 'Diagnostic_in_family_history', 'High_calorie_diet', 'Calorie_monitoring'])
plt.figure(figsize=(18, 6))
for column in other_categorical_attributes.columns:
    plt.subplot(1, len(other_categorical_attributes.columns), list(other_categorical_attributes.columns).index(column) + 1)
    sns.countplot(data=categorical_attributes[column])
    plt.title(f'Histograma pentru {column}')
    plt.xlabel('Valoare')
    plt.ylabel('Contor')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

######################################################
# PARTEA II#
######################################################

# Identificarea valorilor -1 ca valori lipsă și înlocuirea lor cu NaN
df['Weight'] = df['Weight'].replace(-1, float('nan'))

# Tratarea valorilor lipsă în coloana 'Weight' cu mediana
imputer = SimpleImputer(strategy='median')
df['Weight'] = imputer.fit_transform(df[['Weight']])

# Calcularea scorurilor Z pentru fiecare coloană numerică
z_scores = df[numeric_columns].apply(zscore)

# Stabilirea pragului pentru scorurile Z
z_threshold = 5

# Eliminarea valorilor care depășesc pragul specificat pentru scorurile Z
df = df[(z_scores.abs() < z_threshold).all(axis=1)]

# Diagramele de la cerinta I dupa prelucrarea datelor

# Vizualizarea datelor numerice
numeric_attributes = df.select_dtypes(include=['float64', 'int64'])
numeric_stats = numeric_attributes.describe()
print("Statistici pentru atributele numerice prelucrate:\n", numeric_stats.to_string())

# Analize de covarianță
correlation_matrix = numeric_attributes.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corelații prelucrata')
plt.show()

# Identificarea coloanelor categorice
categorical_columns = [col for col in df.columns if col not in numeric_columns]

# Transformarea datelor categorice în valori numerice folosind LabelEncoder
label_encoder = LabelEncoder()

# Dicționar pentru a stoca codificările pentru fiecare coloană
label_encodings = {}

# Aplicarea LabelEncoder pe fiecare coloană categorică
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
    numeric_columns.append(column)  # Adăugarea coloanei la lista numeric_columns
    # Salvarea codificărilor în dicționar
    label_encodings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


numeric_stats = df[numeric_columns].describe()
print("Statistici pentru atribute dupa labelEncoding :\n", numeric_stats.to_string())

# Afișarea codificărilor pentru fiecare coloană
for column, encoding in label_encodings.items():
    print(f"Codificările pentru coloana '{column}':")
    print(encoding)

# Creăm un obiect VarianceThreshold cu pragul specificat (de exemplu, 0.1)
selector = VarianceThreshold(threshold=0.5)
# Aplicăm selectorul pe setul de date pentru a elimina caracteristicile cu variație mică
df_reduced = selector.fit_transform(df[numeric_columns])
# Obținem numele caracteristicilor selectate
selected_features = df[numeric_columns].columns[selector.get_support()]
# Vizualizăm setul de date redus
df_reduced = pd.DataFrame(df_reduced, columns=selected_features)

# Dimensiunile datelor înainte de selecția caracteristicilor
print("Dimensiunile datelor înainte de selecția caracteristicilor:", df.shape)

# Dimensiunile datelor după selecția caracteristicilor
print("Dimensiunile datelor după selecția caracteristicilor:", df_reduced.shape)

numeric_stats = df_reduced.describe()
print("Statistici pentru atributele selectate in urma VarianceThreshold:\n", numeric_stats.to_string())

# Definim setul de caracteristici (X) și etichetele (y)
X = df_reduced.drop('Diagnostic', axis=1)  # Excludem coloana 'Diagnostic' din setul de caracteristici
y = df_reduced['Diagnostic']

# Divizăm datele în seturi de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definim seturile de hiper-parametri pentru fiecare algoritm
svm_params = {'kernel': ['linear', 'rbf', 'sigmoid'],
              'C': [0.1, 1, 10, 100]}

rf_params = {'n_estimators': [50, 100, 200],
             'max_depth': [None, 10, 20],
             'max_features': ['sqrt', 'log2']}

et_params = {'n_estimators': [50, 100, 200],
             'max_depth': [None, 10, 20],
             'max_features': ['sqrt', 'log2']}

gbt_params = {'n_estimators': [50, 100, 200],
              'max_depth': [3, 5, 7],
              'learning_rate': [0.01, 0.1, 0.2]}


# Inițializăm clasa GridSearchCV pentru fiecare algoritm
# Potrivim grilele căutării pe datele noastre
# Obținem cele mai bune scoruri și hiper-parametri pentru fiecare algoritm
# Cronometram aplicarea algoritmului pt toti parametrii pentru a ii compara

start = time.time()
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train, y_train)
end = time.time()
print("Timp pentru SVM:", end - start)
print("Cel mai bun scor pentru SVM:", svm_grid.best_score_)
print("Hiper-parametrii optimi pentru SVM:", svm_grid.best_params_)

start = time.time()
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
end = time.time()
print("Timp pentru RandomForest:", end - start)
print("Cel mai bun scor pentru RandomForest:", rf_grid.best_score_)
print("Hiper-parametrii optimi pentru RandomForest:", rf_grid.best_params_)

start = time.time()
et_grid = GridSearchCV(ExtraTreesClassifier(), et_params, cv=5, scoring='accuracy')
et_grid.fit(X_train, y_train)
end = time.time()
print("Timp pentru ExtraTrees:", end - start)
print("Cel mai bun scor pentru ExtraTrees:", et_grid.best_score_)
print("Hiper-parametrii optimi pentru ExtraTrees:", et_grid.best_params_)

start = time.time()
gbt_grid = GridSearchCV(GradientBoostingClassifier(), gbt_params, cv=5, scoring='accuracy')
gbt_grid.fit(X_train, y_train)
end = time.time()
print("Timp pentru GradientBoostedTrees:", end - start)
print("Cel mai bun scor pentru GradientBoostedTrees:", gbt_grid.best_score_)
print("Hiper-parametrii optimi pentru GradientBoostedTrees:", gbt_grid.best_params_)

# Definim o funcție pentru a calcula metricile cerute
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    return accuracy, precision, recall, f1

# Definim o funcție pentru a afișa metricile într-un tabel
def print_metrics_table(grid_result):
    print("Configurație hiper-parametrii:", grid_result.best_params_)
    print("Medie și varianță pentru metrici:")
    print("Metrică            Medie              Varianță")
    print("------------------------------------------------------")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print(f"Acuratețe          {mean:.4f}             {std:.4f}")

        # Calculăm metricile pentru fiecare clasă în parte
        y_pred = grid_result.best_estimator_.predict(X_test)
        accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
        for i in range(len(precision)):
            print(f"Clasă {i+1}")
            print(f"Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    print("\n")

# Aplicăm funcția pentru fiecare algoritm
print("Rezultate pentru SVM:")
print_metrics_table(svm_grid)

print("Rezultate pentru RandomForest:")
print_metrics_table(rf_grid)

print("Rezultate pentru ExtraTrees:")
print_metrics_table(et_grid)

print("Rezultate pentru GradientBoostedTrees:")
print_metrics_table(gbt_grid)

# Definim o funcție pentru a afișa matricea de confuzie
def plot_confusion_matrix(matrix, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Calculăm și afișăm matricea de confuzie pentru cea mai bună configurație a hiper-parametrilor pentru fiecare algoritm
best_svm = svm_grid.best_estimator_
svm_pred = best_svm.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_pred)
plot_confusion_matrix(svm_cm, classes=['Clasa 1', 'Clasa 2', 'Clasa 3', 'Clasa 4', 'Clasa 5', 'Clasa 6'])

best_rf = rf_grid.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred)
plot_confusion_matrix(rf_cm, classes=['Clasa 1', 'Clasa 2', 'Clasa 3', 'Clasa 4', 'Clasa 5', 'Clasa 6'])

best_et = et_grid.best_estimator_
et_pred = best_et.predict(X_test)
et_cm = confusion_matrix(y_test, et_pred)
plot_confusion_matrix(et_cm, classes=['Clasa 1', 'Clasa 2', 'Clasa 3', 'Clasa 4', 'Clasa 5', 'Clasa 6'])

best_gbt = gbt_grid.best_estimator_
gbt_pred = best_gbt.predict(X_test)
gbt_cm = confusion_matrix(y_test, gbt_pred)
plot_confusion_matrix(gbt_cm, classes=['Clasa 1', 'Clasa 2', 'Clasa 3', 'Clasa 4', 'Clasa 5', 'Clasa 6'])
