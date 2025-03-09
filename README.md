# MolGen: Molecule Generation with GANs

## Overview
MolGen is a GAN-based model to generate novel molecular structures from SMILES strings. It consists of a generator using LSTM to create molecules and a discriminator to assess their validity. The goal is to generate valid and novel molecules for drug discovery and material science.

## Installation

### Dependencies

```bash
pip install torch rdkit matplotlib allennlp

## Usage

### Training the Model

Prepare a dataset of SMILES strings, then train the model:

```python
from molgen import MolGen
data = ["CCO", "CCN", "OCC", "CCOCC"]  # Example data
model = MolGen(data)
train_loader = model.create_dataloader(data, batch_size=128, shuffle=True)
model.train_n_steps(train_loader, max_step=10000, evaluate_every=50)

### Generating Molecules

To generate new molecules after training:

```python
generated_molecules = model.generate_n(10)
for mol in generated_molecules:
    print(mol)

### Evaluation

Evaluate model performance with:

```python
score = model.evaluate_n(100)
print(f"Evaluation score: {score}")

## Model Architecture
**Generator**: LSTM-based model that generates molecular sequences token by token.
**Discriminator**: Evaluates the generated sequences as real or fake using an LSTM encoder.

![image alt](https://github.com/amangeldinaa/MolGen/blob/6e7ec1be1006404feb90c16acfb86fd9eb0b9b90/MolGen-architechture.png)

## Results

- **Avg. Quantum Yield:** 0.8  
- **Training Set Similarity:** 0.6  
- **Novelty:** 0.7  
- **Uniqueness:** 0.9  
- **Validity:** 1.0  

The model demonstrates strong performance in generating valid and unique luminescent molecules, with high novelty and quantum yield scores. The similarity to the training set indicates a balance between innovation and adherence to known molecular structures.

![image alt](https://github.com/amangeldinaa/MolGen/blob/6e7ec1be1006404feb90c16acfb86fd9eb0b9b90/MolGen-results.png)
