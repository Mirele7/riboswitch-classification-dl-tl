# Riboswitch Classification using Deep Learning and Transfer Learning

This repository contains the code, data, and trained models for the classification 
of riboswitch sequences using Graph Neural Networks (GNN) with traditional deep 
learning (DL) and transfer learning (TL) approaches.

## Repository Structure

```
riboswitch-classification-dl-tl/
├── DLmodel/
│   ├── models/        # Trained GNN model weights
│   ├── results_model/ # Classification results
│   ├── src/           # Source code
│   ├── README.md
│   └── requirements.txt
├── TL_model/
│   ├── models/        # Trained GNN model weights (transfer learning)
│   ├── results_model/ # Classification results
│   ├── src/           # Source code
│   ├── README.md
│   └── requirements.txt
├── train_data/        # Training sequences with species and Rfam family annotations
└── test_data/         # Sliding windows generated for each test riboswitch sequence
```

## Datasets

Training data consists of riboswitch sequences from the Rfam v15.1 (2026) database, 
organized into three taxonomic distance datasets:
- **Low**: Trees containing Enterobacteriaceae (*Escherichia coli* pruned)
- **Medium**: Trees containing Gammaproteobacteria (Enterobacteriaceae subtree pruned)
- **High**: Trees not containing Gammaproteobacteria

Test data consists of riboswitch sequences from families RF00059, RF01055, RF00050, 
RF00174, RF00168, and RF01056, annotated in the complete genome of *E. coli* str. 
K-12 substr. MG1655 (NC_000913.3), using genomic sliding windows of 200 nt upstream 
and downstream of each annotated riboswitch.

## Models

Two GNN-based classification models are provided:
- **DLmodel**: Traditional deep learning model trained from scratch
- **TLmodel**: Transfer learning model fine-tuned from the DL model

## Requirements

See `requirements.txt` inside each model directory for dependencies.
