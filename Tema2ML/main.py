import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


##### PARTEA I #####

# Încărcăm seturile de date pentru bătăile de inimă anormale și normale
abnormal_data = pd.read_csv("ptbdb_abnormal.csv", header=None)
normal_data = pd.read_csv("ptbdb_normal.csv", header=None)

# Renunțăm la numele coloanelor pentru a reflecta faptul că eticheta este ultima coloană
abnormal_data.columns = [f'feature_{i}' if i != 187 else 'label' for i in range(188)]
normal_data.columns = [f'feature_{i}' if i != 187 else 'label' for i in range(188)]

# Analiza echilibrului de clase pentru bătăile de inimă anormale
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=abnormal_data)
plt.title('Frecvența de apariție a claselor în setul de date pentru bătăi de inimă anormale')
plt.xlabel('Clasă')
plt.ylabel('Număr de apariții')
plt.show()

# Analiza echilibrului de clase pentru bătăile de inimă normale
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=normal_data)
plt.title('Frecvența de apariție a claselor în setul de date pentru bătăi de inimă normale')
plt.xlabel('Clasă')
plt.ylabel('Număr de apariții')
plt.show()

# Vizualizarea unui exemplu de serie pentru fiecare categorie de aritmie din setul de date PTB
def plot_example_series(data, title):
    plt.figure(figsize=(14, 6))
    unique_labels = data['label'].unique()
    for i, label in enumerate(unique_labels):
        plt.subplot(1, len(unique_labels), i + 1)
        example_series = data[data['label'] == label].iloc[0, :-1]
        plt.plot(example_series)
        plt.title(f'Exemplu de serie pentru clasa {label} - {title}')
        plt.xlabel('Timp')
        plt.ylabel('Valoare')
    plt.tight_layout()
    plt.show()

plot_example_series(abnormal_data, "Bătăi de inimă anormale")
plot_example_series(normal_data, "Bătăi de inimă normale")

# Calculăm media și deviația standard per unitate de timp pentru fiecare clasă de aritmie
mean_per_class_abnormal = abnormal_data.groupby('label').mean().T
mean_per_class_normal = normal_data.groupby('label').mean().T
std_per_class_abnormal = abnormal_data.groupby('label').std().T
std_per_class_normal = normal_data.groupby('label').std().T

# Afișăm graficul pentru media per clasă de aritmie pentru bătăile de inimă anormale
plt.figure(figsize=(12, 6))
for label in mean_per_class_abnormal.columns:
    plt.plot(mean_per_class_abnormal.index, mean_per_class_abnormal[label], label=f'Clasa {label}')
plt.title('Medie per clasă de aritmie pentru bătăile de inimă anormale')
plt.xlabel('Timp')
plt.ylabel('Medie')
plt.legend()
plt.show()

# Afișăm graficul pentru deviația standard per clasă de aritmie pentru bătăile de inimă anormale
plt.figure(figsize=(12, 6))
for label in std_per_class_abnormal.columns:
    plt.plot(std_per_class_abnormal.index, std_per_class_abnormal[label], label=f'Clasa {label}')
plt.title('Deviație standard per clasă de aritmie pentru bătăile de inimă anormale')
plt.xlabel('Timp')
plt.ylabel('Deviație standard')
plt.legend()
plt.show()

# Afișăm graficul pentru media per clasă de aritmie pentru bătăile de inimă normale
plt.figure(figsize=(12, 6))
for label in mean_per_class_normal.columns:
    plt.plot(mean_per_class_normal.index, mean_per_class_normal[label], label=f'Clasa {label}')
plt.title('Medie per clasă de aritmie pentru bătăile de inimă normale')
plt.xlabel('Timp')
plt.ylabel('Medie')
plt.legend()
plt.show()

# Afișăm graficul pentru deviația standard per clasă de aritmie pentru bătăile de inimă normale
plt.figure(figsize=(12, 6))
for label in std_per_class_normal.columns:
    plt.plot(std_per_class_normal.index, std_per_class_normal[label], label=f'Clasa {label}')
plt.title('Deviație standard per clasă de aritmie pentru bătăile de inimă normale')
plt.xlabel('Timp')
plt.ylabel('Deviație standard')
plt.legend()
plt.show()

############################## PARTEA II ##############################

######## Pregătirea datelor ########

######### Preprocesarea ca in Tema 1 a setului de date Patients ##########

# Încărcăm setul de date
patients_data = pd.read_csv("date_tema_1_iaut_2024.csv", decimal=',')

# Conversia coloanelor numerice în format float
numeric_columns = ['Regular_fiber_diet', 'Sedentary_hours_daily', 'Age', 'Est_avg_calorie_intake',
                   'Main_meals_daily', 'Height', 'Water_daily', 'Weight', 'Physical_activity_level',
                   'Technology_time_use']

for column in numeric_columns:
    patients_data[column] = pd.to_numeric(patients_data[column], errors='coerce')

# Identificarea valorilor -1 ca valori lipsă și înlocuirea lor cu NaN
patients_data['Weight'] = patients_data['Weight'].replace(-1, float('nan'))

# Tratarea valorilor lipsă în coloana 'Weight' cu mediana
imputer = SimpleImputer(strategy='median')
patients_data['Weight'] = imputer.fit_transform(patients_data[['Weight']])

# Calcularea scorurilor Z pentru fiecare coloană numerică
z_scores = patients_data[numeric_columns].apply(zscore)

# Stabilirea pragului pentru scorurile Z
z_threshold = 5

# Eliminarea valorilor care depășesc pragul specificat pentru scorurile Z
patients_data = patients_data[(z_scores.abs() < z_threshold).all(axis=1)]

# Identificarea coloanelor categorice
categorical_columns = [col for col in patients_data.columns if col not in numeric_columns]

# Transformarea datelor categorice în valori numerice folosind LabelEncoder
label_encoder = LabelEncoder()

# Dicționar pentru a stoca codificările pentru fiecare coloană
label_encodings = {}

# Aplicarea LabelEncoder pe fiecare coloană categorică
for column in categorical_columns:
    patients_data[column] = label_encoder.fit_transform(patients_data[column])
    numeric_columns.append(column)  # Adăugarea coloanei la lista numeric_columns
    # Salvarea codificărilor în dicționar
    label_encodings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Separăm etichetele de caracteristici
X_patients = patients_data.drop(columns=['Diagnostic'])
y_patients = patients_data['Diagnostic']

# Verificăm dacă etichetele sunt indexate corect de la 0 la n-1, si folosim apoi acest mapping
unique_labels = sorted(y_patients.unique())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
y_patients = y_patients.map(label_mapping)

# Împărțirea setului de date în seturi de antrenare și testare
X_patients_train, X_patients_test, y_patients_train, y_patients_test = train_test_split(X_patients, y_patients, test_size=0.2, random_state=42)

# Standardizarea datelor
scaler = StandardScaler()
X_patients_train = scaler.fit_transform(X_patients_train)
X_patients_test = scaler.transform(X_patients_test)

########## Pregătirea setului de date Patients ###########

# Conversie la tensori PyTorch
X_patients_train_tensor = torch.tensor(X_patients_train, dtype=torch.float32)
y_patients_train_tensor = torch.tensor(y_patients_train.values, dtype=torch.long)
X_patients_test_tensor = torch.tensor(X_patients_test, dtype=torch.float32)
y_patients_test_tensor = torch.tensor(y_patients_test.values, dtype=torch.long)

# Creăm DataLoader pentru setul de date Patients
train_dataset_patients = TensorDataset(X_patients_train_tensor, y_patients_train_tensor)
test_dataset_patients = TensorDataset(X_patients_test_tensor, y_patients_test_tensor)

train_loader_patients = DataLoader(train_dataset_patients, batch_size=32, shuffle=True)
test_loader_patients = DataLoader(test_dataset_patients, batch_size=32, shuffle=False)


# Pregătirea setului de date PTB Diagnostic ECG
ptb_data = pd.concat([abnormal_data, normal_data], axis=0)
X_ptb = ptb_data.iloc[:, :-1].values
y_ptb = ptb_data.iloc[:, -1].values

X_ptb_train, X_ptb_test, y_ptb_train, y_ptb_test = train_test_split(X_ptb, y_ptb, test_size=0.2, random_state=42)

# Conversie la tensori PyTorch
X_ptb_train_tensor = torch.tensor(X_ptb_train, dtype=torch.float32)
y_ptb_train_tensor = torch.tensor(y_ptb_train, dtype=torch.long)
X_ptb_test_tensor = torch.tensor(X_ptb_test, dtype=torch.float32)
y_ptb_test_tensor = torch.tensor(y_ptb_test, dtype=torch.long)

# Creăm DataLoader pentru setul de date PTB
train_dataset_ptb = TensorDataset(X_ptb_train_tensor, y_ptb_train_tensor)
test_dataset_ptb = TensorDataset(X_ptb_test_tensor, y_ptb_test_tensor)

train_loader_ptb = DataLoader(train_dataset_ptb, batch_size=32, shuffle=True)
test_loader_ptb = DataLoader(test_dataset_ptb, batch_size=32, shuffle=False)


######## Arhitectura de tip MLP pentru setul de date Patients ########

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Funcție de antrenare
def train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []
    test_losses = []
    all_labels = []
    all_preds = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Antrenare
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0

        # Testare
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    print(f'Accuracy: {accuracy * 100}%')
    return train_losses, test_losses, accuracy, precision, recall, f1, all_labels, all_preds


# Configurații pentru arhitecturile MLP
mlp_hidden_sizes = [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64, 32]]
results_patients = {}
results_ptb = {}
num_epochs = 20

# Antrenare și testare pentru Patients
for hidden_sizes in mlp_hidden_sizes:
    print(f"MLP for patients with hidden sizes: {hidden_sizes}")
    input_size = X_patients_train.shape[1]
    output_size = len(label_mapping)
    model = MLP(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy, precision, recall, f1, all_labels, all_preds = train_and_test(model, train_loader_patients,
                                                                                test_loader_patients, criterion,
                                                                                optimizer, num_epochs)
    results_patients[f"Hidden sizes: {hidden_sizes}"] = {"train": train_losses, "test": test_losses,
                                                         "accuracy": accuracy,
                                                         "precision": precision, "recall": recall, "f1": f1}

# Antrenare și testare pentru PTB Diagnostic ECG
for hidden_sizes in mlp_hidden_sizes:
    print(f"MLP for PTB Diagnostic ECG with hidden sizes: {hidden_sizes}")
    input_size = X_ptb_train.shape[1]
    output_size = 2
    model = MLP(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy, precision, recall, f1, all_labels, all_preds = train_and_test(
        model, train_loader_ptb, test_loader_ptb, criterion, optimizer, num_epochs)
    results_ptb[f"Hidden sizes: {hidden_sizes}"] = {"train": train_losses, "test": test_losses, "accuracy": accuracy,
                                                    "precision": precision, "recall": recall, "f1": f1,
                                                    "labels": all_labels, "preds": all_preds}

# Afișarea graficului cu curbele de loss pentru Patients
plt.figure(figsize=(10, 6))
for name, losses in results_patients.items():
    train_losses = losses["train"]
    test_losses = losses["test"]
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label=f"{name} - Train Loss")
    plt.plot(epochs, test_losses, label=f"{name} - Test Loss")

plt.title("Curbele de Loss pentru antrenare și testare - Patients")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Afișarea graficului cu curbele de loss pentru PTB Diagnostic ECG
plt.figure(figsize=(10, 6))
for name, losses in results_ptb.items():
    train_losses = losses["train"]
    test_losses = losses["test"]
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label=f"{name} - Train Loss")
    plt.plot(epochs, test_losses, label=f"{name} - Test Loss")

plt.title("Curbele de Loss pentru antrenare și testare - PTB Diagnostic ECG")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

######### Arhitectură de tip convoluțională #########

# Definirea arhitecturii modelului
# Definim arhitectura ConvNet

# Define your ConvNet class with the correct number of input channels
class ConvNet(nn.Module):
    def __init__(self, kernel_size=5):
        super(ConvNet, self).__init__()

        # Primul strat convoluțional
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Al doilea strat convoluțional
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculăm dimensiunea de ieșire după straturile convoluționale pentru lungimea maximă a secvențelor
        def conv_output_size(size, kernel_size=5, stride=1, padding=2, dilation=1):
            return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        conv_output = conv_output_size(188, kernel_size=kernel_size, padding=kernel_size//2)
        conv_output = conv_output_size(conv_output, kernel_size=kernel_size, padding=kernel_size//2, stride=2)
        conv_output = conv_output_size(conv_output, kernel_size=kernel_size, padding=kernel_size//2, stride=2)

        # Strat liniar final
        self.fc = nn.Linear(128 * conv_output, 2)

    def forward(self, x):
        # Aplicăm padding la secvențele mai scurte pentru a le aduce la lungimea maximă
        x = F.pad(x, (0, 188 - x.size(-1)))

        # Adăugăm o dimensiune pentru canalele de intrare
        x = x.unsqueeze(1)

        # Aplicăm straturile convoluționale și liniare
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


kernel_sizes = [3, 5, 7]
results_ConvNet = {}

# Antrenare și testare pentru fiecare dimensiune a kernel-ului
for kernel_size in kernel_sizes:
    print(f"Experiment with ConvNet kernel size: {kernel_size}")
    model_ptb_cnn = ConvNet(kernel_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ptb_cnn.parameters(), lr=0.001)
    train_losses, test_losses, accuracy, precision, recall, f1, all_labels, all_preds = train_and_test(
        model_ptb_cnn, train_loader_ptb, test_loader_ptb, criterion, optimizer, num_epochs)
    results_ConvNet[f"Kernel size: {kernel_size}"] = {"train": train_losses, "test": test_losses, "accuracy": accuracy,
                                                      "precision": precision, "recall": recall, "f1": f1,
                                                      "labels": all_labels, "preds": all_preds}

# Afișarea graficului cu curbele de loss
plt.figure(figsize=(10, 6))
for name, losses in results_ConvNet.items():
    train_losses = losses["train"]
    test_losses = losses["test"]
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label=f"{name} - Train Loss")
    plt.plot(epochs, test_losses, label=f"{name} - Test Loss")

plt.title("Curbele de Loss pentru antrenare și testare")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Crearea DataFrame-urilor pentru rezultate
df_mlp_patients = pd.DataFrame.from_dict(results_patients, orient='index')
df_mlp_ptb = pd.DataFrame.from_dict(results_ptb, orient='index')
df_ConvNet_ptb = pd.DataFrame.from_dict(results_ConvNet, orient='index')

# Afișarea DataFrame-urilor
print("\nResults for MLP - Patients")
print(df_mlp_patients.to_string())

print("\nResults for MLP - PTB Diagnostic ECG")
print(df_mlp_ptb.to_string())

print("\nResults for ConvNet - PTB Diagnostic ECG")
print(df_ConvNet_ptb.to_string())

# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, title):
    conf_matrix = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Abnormal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# Plot confusion matrices for MLP models
for hidden_sizes, result in results_ptb.items():
    plot_confusion_matrix(result["labels"], result["preds"], f"Confusion Matrix for MLP with hidden sizes: {hidden_sizes}")

# Plot confusion matrices for ConvNet models
for kernel_size, result in results_ConvNet.items():
    plot_confusion_matrix(result["labels"], result["preds"], f"Confusion Matrix for ConvNet with kernel size: {kernel_size}")