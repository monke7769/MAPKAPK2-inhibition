import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import LayeredFingerprint
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import cross_val_score

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.base import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

from keras.models import load_model
fcfpnet = load_model('trained_models\\fcfpnet.keras')
ecfpnet = load_model('trained_models\\ecfpnet.keras')
featnet = load_model('trained_models\\featnet.keras')
laynet = load_model('trained_models\\laynet.keras')
mornet = load_model('trained_models\\mornet.keras')
p2gobnet = load_model('trained_models\\p2gobnet.keras')
p2mapnet = load_model('trained_models\\p2mapnet.keras')
patnet = load_model('trained_models\\patnet.keras')
secfpnet = load_model('trained_models\\secfpnet.keras')
tornet = load_model('trained_models\\tornet.keras')

# Path to compound set for evaluation
file_path = 'data\\20230810_in_thermo_NCI_chemdiv.smi' # in-house "in-thermo" set

# Read the .smi file
df = pd.read_csv(file_path, sep='\t', header=None, names=['Compound'])

# Feature Creation
smiles = df.Compound
molecules = [Chem.MolFromSmiles(s.split(",")[0]) for s in smiles]

transformer = FPVecTransformer(kind='fcfp', dtype=float)
fcfp = transformer(molecules)

transformer = FPVecTransformer(kind='ecfp-count', dtype=float)
ecfp = transformer(molecules)

transformer = FPVecTransformer(kind='secfp', dtype=float)
secfp = transformer(molecules)

transformer = MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float)
p2gob = transformer(molecules)

featfp = []
for mol in molecules:
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useFeatures=True)
    fp_array = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
    featfp.append(fp_array)
featdf = pd.DataFrame(featfp)

transformer = MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float)
p2map = transformer(molecules)

morfp = []
for mol in molecules:
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo={})
    fp_array = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
    morfp.append(fp_array)
mordf = pd.DataFrame(morfp)

transformer = FPVecTransformer(kind='pattern', dtype=float)
patfp = transformer(molecules)

torfp = []
for mol in molecules:
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
    fp_array = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
    torfp.append(fp_array)
tordf = pd.DataFrame(torfp)

layfp = []
for mol in molecules:
    fp = LayeredFingerprint(mol)
    fp_array = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
    layfp.append(fp_array)
laydf = pd.DataFrame(layfp)

# testing

binary = np.round(fcfpnet.predict(fcfp)) # This gives the class predictions (0 vs 1)
prob = fcfpnet.predict(fcfp).ravel()  # This gives the probabilities for each class
fcfp_binary_df = pd.DataFrame(binary, columns=['fcfp_binary'])
fcfp_prob_df = pd.DataFrame(prob, columns=['fcfp_prob'])

binary = np.round(ecfpnet.predict(ecfp))
prob = ecfpnet.predict(ecfp).ravel()
ecfp_binary_df = pd.DataFrame(binary, columns=['ecfp_binary'])
ecfp_prob_df = pd.DataFrame(prob, columns=['ecfp_prob'])

binary = np.round(secfpnet.predict(secfp))
prob = secfpnet.predict(secfp).ravel()
secfp_binary_df = pd.DataFrame(binary, columns=['secfp_binary'])
secfp_prob_df = pd.DataFrame(prob, columns=['secfp_prob'])

binary = np.round(p2gobnet.predict(p2gob))
prob = p2gobnet.predict(p2gob).ravel()
p2gob_binary_df = pd.DataFrame(binary, columns=['p2gob_binary'])
p2gob_prob_df = pd.DataFrame(prob, columns=['p2gob_prob'])

binary = np.round(featnet.predict(featdf))
prob = featnet.predict(featdf).ravel()
feat_binary_df = pd.DataFrame(binary, columns=['feat_binary'])
feat_prob_df = pd.DataFrame(prob, columns=['feat_prob'])

binary = np.round(p2mapnet.predict(p2map))
prob = p2mapnet.predict(p2map).ravel()
p2map_binary_df = pd.DataFrame(binary, columns=['p2map_binary'])
p2map_prob_df = pd.DataFrame(prob, columns=['p2map_prob'])

binary = np.round(mornet.predict(mordf))
prob = mornet.predict(mordf).ravel()
mor_binary_df = pd.DataFrame(binary, columns=['mor_binary'])
mor_prob_df = pd.DataFrame(prob, columns=['mor_prob'])

binary = np.round(patnet.predict(patfp))
prob = patnet.predict(patfp).ravel()
pat_binary_df = pd.DataFrame(binary, columns=['pat_binary'])
pat_prob_df = pd.DataFrame(prob, columns=['pat_prob'])

binary = np.round(tornet.predict(tordf))
prob = tornet.predict(tordf).ravel()
tor_binary_df = pd.DataFrame(binary, columns=['tor_binary'])
tor_prob_df = pd.DataFrame(prob, columns=['tor_prob'])

binary = np.round(laynet.predict(laydf))
prob = laynet.predict(laydf).ravel()
lay_binary_df = pd.DataFrame(binary, columns=['lay_binary'])
lay_prob_df = pd.DataFrame(prob, columns=['lay_prob'])

combined_df = pd.concat([fcfp_binary_df,fcfp_prob_df,ecfp_binary_df,ecfp_prob_df,p2gob_binary_df,p2gob_prob_df,secfp_binary_df,secfp_prob_df,feat_binary_df,feat_prob_df,p2map_binary_df,p2map_prob_df,mor_binary_df,mor_prob_df,pat_binary_df,pat_prob_df,tor_binary_df,tor_prob_df,lay_binary_df,lay_prob_df], axis=1)

combined_df['sum_binary'] = combined_df['fcfp_binary'] + combined_df['ecfp_binary'] + combined_df['secfp_binary'] + combined_df['p2gob_binary'] + combined_df['feat_binary'] + combined_df['p2map_binary'] + combined_df['mor_binary'] + combined_df['pat_binary'] + combined_df['tor_binary'] + combined_df['lay_binary']
combined_df['sum_prob'] = combined_df['fcfp_prob'] + combined_df['ecfp_prob'] + combined_df['secfp_prob'] + combined_df['p2gob_prob'] + combined_df['feat_prob'] + combined_df['p2map_prob'] + combined_df['mor_prob'] + combined_df['pat_prob'] + combined_df['tor_prob'] + combined_df['lay_prob']

combined_df.to_csv('inthermo_results.csv') 