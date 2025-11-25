import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
import os


#ARQUIVOS
ATTRIBUTES_FILE = './dataset/attributes.csv'
LABELS_FILE = './dataset/label.csv'          

CLASS_NAMES = {
    0: 'Normal',
    1: 'Charge',
    2: 'Discharge',
    3: 'Friction',
    4: 'Charge Discharge',
    5: 'Charge Friction',
    6: 'Discharge Friction',
    7: 'Charge Discharge Friction'
}

# Configuração para os dados aleatorios se repetirem
np.random.seed(42)
tf.random.set_seed(42)

def carrega_dados(attr_caminho, lbl_caminho):

    print(f"Lendo atributos: {attr_caminho}")
    print(f"Lendo labels: {lbl_caminho}")
    
    # Tenta ler assumindo que tem header padrão
    X_df = pd.read_csv(attr_caminho)
    y_df = pd.read_csv(lbl_caminho)

    # converte para vetor
    X = X_df.values
    y = y_df.values.flatten() 

    num_classes = len(np.unique(y))
    num_features = X.shape[1]
    
    print(f"Dataset carregado com sucesso:")
    print(f" - Amostras: {X.shape[0]}")
    print(f" - Features (Timestamps): {num_features}")
    print(f" - Classes únicas encontradas: {num_classes} {np.unique(y)}")

    # One-hot para redes neurais
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    return X, y_categorical, y, num_features, num_classes

# MODELOS (Tópico 3.4)

def constroi_mlp(config_name, shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=shape))
    
    if config_name == '4N':
        model.add(layers.Dense(4, activation='relu'))
    elif config_name == '8N':
        model.add(layers.Dense(8, activation='relu'))
    elif config_name == '16N':
        model.add(layers.Dense(16, activation='relu'))
    elif config_name == '32N':
        model.add(layers.Dense(32, activation='relu'))
    elif config_name == '16-8N':
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def constroi_cnn(config_name, shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=shape))
    
    if config_name == 'M1':
        model.add(layers.Conv1D(filters=1, kernel_size=8, padding='same'))
        model.add(layers.AveragePooling1D(pool_size=4))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        
    elif config_name == 'M2':
        model.add(layers.Conv1D(filters=2, kernel_size=8, padding='same'))
        model.add(layers.AveragePooling1D(pool_size=4))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        
    elif config_name == 'M3':
        model.add(layers.Conv1D(filters=1, kernel_size=16, padding='same'))
        model.add(layers.AveragePooling1D(pool_size=4))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        
    elif config_name == 'M4':
        model.add(layers.Conv1D(filters=1, kernel_size=8, padding='same'))
        model.add(layers.AveragePooling1D(pool_size=8))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        
    elif config_name == 'M5':
        model.add(layers.Conv1D(filters=1, kernel_size=8, padding='same'))
        model.add(layers.AveragePooling1D(pool_size=4))
        model.add(layers.Conv1D(filters=1, kernel_size=8, padding='same'))
        model.add(layers.AveragePooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# EXPERIMENTOS
def run_experiments():
        
    X, y_cat, y_integers, n_features, n_classes = carrega_dados(ATTRIBUTES_FILE, LABELS_FILE)
    
    # divisao treino/teste (10% Teste)
    X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
        X, y_cat, y_integers, test_size=0.10, stratify=y_integers, random_state=42
    )
    
    # CNN: (samples, timesteps, features=1)
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    mlp_shape = (n_features,)
    cnn_shape = (n_features, 1)

    EPOCHS = 50 
    BATCH_SIZE = 32
    K_FOLDS = 10
    
    resultados = []
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    print(f"\nIniciando Validação Cruzada (K={K_FOLDS})...")
    
    # MLP
    confiiguracoes_mlp = ['4N', '8N', '16N', '32N', '16-8N']
    for cfg in confiiguracoes_mlp:
        print(f"Avaliando MLP: {cfg}")
        rel_acuracia = []
        for train_idx, val_idx in skf.split(X_train, y_train_int):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = constroi_mlp(cfg, mlp_shape, n_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_fold_train, y_fold_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            _, acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            rel_acuracia.append(acc)
            
        mean_acc = np.mean(rel_acuracia)
        std_acc = np.std(rel_acuracia)
        resultados.append({'Model': 'MLP', 'Config': cfg, 'Val_Acc_Mean': mean_acc, 'Val_Acc_Std': std_acc})
        print(f"  -> Acc Média: {mean_acc:.4f}")

    # CNN
    configuracoes_cnn = ['M1', 'M2', 'M3', 'M4', 'M5']
    for cfg in configuracoes_cnn:
        print(f"Avaliando CNN: {cfg}")
        rel_acuracia = []
        for train_idx, val_idx in skf.split(X_train, y_train_int):
            X_fold_train, X_fold_val = X_train_cnn[train_idx], X_train_cnn[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = constroi_cnn(cfg, cnn_shape, n_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_fold_train, y_fold_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            _, acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            rel_acuracia.append(acc)
            
        mean_acc = np.mean(rel_acuracia)
        std_acc = np.std(rel_acuracia)
        resultados.append({'Model': 'CNN', 'Config': cfg, 'Val_Acc_Mean': mean_acc, 'Val_Acc_Std': std_acc})
        print(f"  -> Acc Média: {mean_acc:.4f}")

    # KNN
    k_values = [1, 2, 5, 10, 20]
    print("\nAvaliando KNN...")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        rel_acuracia = []
        for train_idx, val_idx in skf.split(X_train, y_train_int):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train_labels = y_train_int[train_idx]
            y_fold_val_labels = y_train_int[val_idx]
            knn.fit(X_fold_train, y_fold_train_labels)
            acc = knn.score(X_fold_val, y_fold_val_labels)
            rel_acuracia.append(acc)
        
        mean_acc = np.mean(rel_acuracia)
        std_acc = np.std(rel_acuracia)
        resultados.append({'Model': 'KNN', 'Config': f'K={k}', 'Val_Acc_Mean': mean_acc, 'Val_Acc_Std': std_acc})
        print(f"  -> KNN K={k}: {mean_acc:.4f}")

    # Resultado Final
    resultados_df = pd.DataFrame(resultados)
    melhor_rodada = resultados_df.loc[resultados_df['Val_Acc_Mean'].idxmax()]
    
    print("\n" + "="*40)
    print(f"MELHOR MODELO: {melhor_rodada['Model']} ({melhor_rodada['Config']})")
    print("="*40)
    
    final_acc = 0
    if melhor_rodada['Model'] == 'MLP':
        modelo_final = constroi_mlp(melhor_rodada['Config'], mlp_shape, n_classes)
        modelo_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        modelo_final.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        loss, final_acc = modelo_final.evaluate(X_test, y_test, verbose=0)
    elif melhor_rodada['Model'] == 'CNN':
        modelo_final = constroi_cnn(melhor_rodada['Config'], cnn_shape, n_classes)
        modelo_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        modelo_final.fit(X_train_cnn, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        loss, final_acc = modelo_final.evaluate(X_test_cnn, y_test, verbose=0)
    elif melhor_rodada['Model'] == 'KNN':
        k = int(melhor_rodada['Config'].split('=')[1])
        modelo_final = KNeighborsClassifier(n_neighbors=k)
        modelo_final.fit(X_train, y_train_int)
        final_acc = modelo_final.score(X_test, y_test_int)

    print(f"Acurácia Final no Teste (10%): {final_acc:.4f}")
    print("\nResumo Completo:")
    print(resultados_df[['Model', 'Config', 'Val_Acc_Mean', 'Val_Acc_Std']])

if __name__ == "__main__":
    run_experiments()