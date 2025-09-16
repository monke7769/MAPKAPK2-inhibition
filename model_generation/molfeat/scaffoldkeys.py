import pandas as pd
import numpy as np
from rdkit import Chem
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import clone_model

# Load in data
data_path = 'data.csv'
url = 'https://drive.google.com/uc?id=1CgZwuxndc0jte8Ns20MYCp3J9YtBlwPX'
gdown.download(url, data_path, quiet=False)
data = pd.read_csv(data_path)
data.rename(columns={'SMILES (Canonical)': 'smiles'}, inplace = True)

def change2binary(letter):
  if letter == 'p':
    return 1
  elif letter == 'n':
    return 0

data['Active'] = data['Active'].apply(change2binary)

from molfeat.trans.base import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer

smiles = data.smiles

molecules = [Chem.MolFromSmiles(s) for s in smiles]

transformer = FPVecTransformer(kind='scaffoldkeys', dtype=float)
features = transformer(molecules)
testfeatures = pd.DataFrame(features)

X = testfeatures
y = data['Active']

X, X_special, y, y_special = train_test_split(X, y, test_size=.1, random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scale the features
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)
X_special_scaled = scaler.transform(X_special)

# Convert the scaled NumPy arrays back to pandas DataFrames
X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
X_special_scaled = pd.DataFrame(X_special_scaled, index=X_special.index, columns=X_special.columns)

# load in ACD set

pat = 'acd.csv'
acd_url = 'https://drive.google.com/uc?id=10UW5yHdHC7riRAM-s-8FhswbuctTYSaz'
gdown.download(acd_url, pat, quiet=False)
acd990 = pd.read_csv(pat)

def all0(smiles):
  if smiles:
    return 0

acd990['Active'] = acd990['Smiles'].apply(all0)

acdsmiles = acd990.Smiles

acd_mol = [Chem.MolFromSmiles(s) for s in acdsmiles]

transformer = FPVecTransformer(kind='scaffoldkeys', dtype=float)
features = transformer(acd_mol)

X_acd = pd.DataFrame(features)
y_acd = acd990['Active']

X_acd_scaled = scaler.transform(X_acd)

# Convert the scaled NumPy arrays back to pandas DataFrames
X_acd_scaled = pd.DataFrame(X_acd_scaled, index=X_acd.index, columns=X_acd.columns)

crossval = StratifiedKFold(n_splits=5)
keras.utils.set_random_seed(42)

layersizes = [[256,128],[256,64],[256,32],[256,16],[256,128,64],[256,128,32],[256,64,32],[256,64,16],[512,256],[512,128],[512,64],[512,256,128],[512,256,64],[512,128,64],[128,64],[128,32],[128,16],[128,64,32],[128,64,16],[128,32,16],[64,32],[64,16],[64,32,16],[32,16],[32,8],[32,16,8],[32,4],[16,8]]
pos = []
neg = []

# the following lists will contain all the AVERAGE TEST set metrics for each set of layer sizes, using 5-fold cross validation

accs, precisions, recalls, f1s, specs, aps, aucs = [], [], [], [], [], [], []

accs_test, precisions_test, recalls_test, f1s_test, specs_test, aps_test, aucs_test = [], [], [], [], [], [], []

for size in layersizes:
  nnet = Sequential()

  # Add first hidden layer (512 units) with ReLU activation and dropout
  nnet.add(Dense(size[0], activation='relu', input_shape=(189,)))  # Replace input_dim with the number of input features
  nnet.add(Dropout(0.25))  # Dropout rate of 0.25
  for i in size[1:]:
    nnet.add(Dense(i, activation='relu'))
    nnet.add(Dropout(0.25))

  # Add output layer (assuming binary classification, adjust for multi-class if needed)
  nnet.add(Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class classification

  # Compile the model with the Adam optimizer, binary crossentropy loss, and accuracy metric
  nnet.compile(optimizer='adam',
                loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class
                metrics=['accuracy'])

  test_acc, train_acc = [], []
  test_precision, train_precision = [], []
  test_recall, train_recall = [], []
  test_f1, train_f1 = [], []
  test_specificity, train_specificity = [], []
  test_ap, train_ap = [], []
  test_auc, train_auc = [], []

  for train_index, test_index in crossval.split(X_scaled, y):
      X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      # clone to ensure each time is independent
      early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

      # Clone and compile the Keras model
      model = clone_model(nnet)
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

      # Train the model with early stopping
      model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=1, callbacks=[early_stopping])

      # Making predictions
      y_pred_train = np.round(model.predict(X_train))
      y_pred_test = np.round(model.predict(X_test))

      y_pred_proba_train = model.predict(X_train).ravel()
      y_pred_proba_test = model.predict(X_test).ravel()

      # Calculating metrics
      test_acc.append(accuracy_score(y_test, y_pred_test))
      train_acc.append(accuracy_score(y_train, y_pred_train))

      test_precision.append(precision_score(y_test, y_pred_test, average='macro'))
      train_precision.append(precision_score(y_train, y_pred_train, average='macro'))

      test_recall.append(recall_score(y_test, y_pred_test, average='macro'))
      train_recall.append(recall_score(y_train, y_pred_train, average='macro'))

      test_f1.append(f1_score(y_test, y_pred_test, average='macro'))
      train_f1.append(f1_score(y_train, y_pred_train, average='macro'))

      tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
      test_specificity.append(tn / (tn + fp))

      tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
      train_specificity.append(tn_train / (tn_train + fp_train))

      test_ap.append(average_precision_score(y_test, y_pred_proba_test))
      train_ap.append(average_precision_score(y_train, y_pred_proba_train))

      test_auc.append(roc_auc_score(y_test, y_pred_proba_test))
      train_auc.append(roc_auc_score(y_train, y_pred_proba_train))
  print(size)
  print('---')
  print("Mean Test Accuracy:", np.mean(test_acc))
  accs.append(np.mean(test_acc))
  print("Mean Test Precision:", np.mean(test_precision))
  precisions.append(np.mean(test_precision))
  print("Mean Test Recall:", np.mean(test_recall))
  recalls.append(np.mean(test_recall))
  print("Mean Test F1-Score:", np.mean(test_f1))
  f1s.append(np.mean(test_f1))
  print("Mean Test Specificity:", np.mean(test_specificity))
  specs.append(np.mean(test_specificity))
  print("Mean Test AP:", np.mean(test_ap))
  aps.append(np.mean(test_ap))
  print("Mean Test AUC:", np.mean(test_auc))
  aucs.append(np.mean(test_auc))
  # for just one set of layer sizes
  print("Mean Train Accuracy:", np.mean(train_acc))
  print("Mean Train Precision:", np.mean(train_precision))
  print("Mean Train Recall:", np.mean(train_recall))
  print("Mean Train F1-Score:", np.mean(train_f1))
  print("Mean Train Specificity:", np.mean(train_specificity))
  print("Mean Train AP:", np.mean(train_ap))
  print("Mean Train AUC:", np.mean(train_auc))
  print('\n')

  # Final test for 0.1 test set and ACD FP
  nnet = Sequential()

  nnet.add(Dense(size[0], activation='relu', input_shape=(189,)))  # Replace input_dim with the number of input features
  nnet.add(Dropout(0.25))  # Dropout rate of 0.25
  for i in size[1:]:
    nnet.add(Dense(i, activation='relu'))
    nnet.add(Dropout(0.25))
  nnet.add(Dense(1, activation='sigmoid'))
  nnet.compile(optimizer=Adam(),
                loss='binary_crossentropy',
                metrics=['accuracy'])
  nnet.fit(X_scaled, y.values.ravel(), validation_split=0.2, epochs=100, verbose=1, callbacks=[early_stopping])
  y_special_pred = np.round(nnet.predict(X_special))
  y_special_pred_proba = nnet.predict(X_special).ravel()

  accs_test.append(accuracy_score(y_special, y_special_pred))
  precisions_test.append(precision_score(y_special, y_special_pred, average='macro'))
  recalls_test.append(recall_score(y_special, y_special_pred, average='macro'))
  f1s_test.append(f1_score(y_special, y_special_pred, average='macro'))
  tn, fp, fn, tp = confusion_matrix(y_special, y_special_pred).ravel()
  specs_test.append(tn / (tn + fp))
  aps_test.append(average_precision_score(y_special, y_special_pred_proba))
  aucs_test.append(roc_auc_score(y_special, y_special_pred_proba))

  y_pred_acd = np.round(nnet.predict(X_acd_scaled))
  pos.append(np.sum(y_pred_acd == 1))  # Count of positives
  neg.append(np.sum(y_pred_acd == 0))  # Count of negatives

categories = ['[256,128]','[256,64]','[256,32]','[256,16]','[256,128,64]','[256,128,32]','[256,64,32]','[256,64,16]','[512,256]','[512,128]','[512,64]','[512,256,128]','[512,256,64]','[512,128,64]','[128,64]','[128,32]','[128,16]','[128,64,32]','[128,64,16]','[128,32,16]','[64,32]','[64,16]','[64,32,16]','[32,16]','[32,8]','[32,16,8]','[32,4]','[16,8]']

with open('featurename.csv', 'w') as file: # AVERAGES USING 5-FOLD CROSSVAL
    # Write the header with commas as the delimiter
    file.write('Index,Model,Network,ACC,Precision,Recall,F1,Specificity,AP,AUC,ACD990 N,ACD990 P\n')
    for i in range(28):
        file.write(f'{i+1},"Feature Name","{categories[i]}",{accs[i]},{precisions[i]},{recalls[i]},{f1s[i]},{specs[i]},{aps[i]},{aucs[i]},{neg[i]},{pos[i]}\n') # need neg and pos up herre (get from bottom)

with open('featurename0.1TEST.csv', 'w') as file: # UNSEEN TEST DATA
    # Write the header with commas as the delimiter
    file.write('Index,Model,Network,ACC,Precision,Recall,F1,Specificity,AP,AUC,ACD990 N,ACD990 P\n')
    for i in range(28):
        file.write(f'{i+1},"Feature Name","{categories[i]}",{accs_test[i]},{precisions_test[i]},{recalls_test[i]},{f1s_test[i]},{specs_test[i]},{aps_test[i]},{aucs_test[i]},{neg[i]},{pos[i]}\n')