# Python for BioInformatics

## Table of Contents
- [Introduction](#introduction)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Python Basics](#python-basics)
- [Data Types and Structures](#data-types-and-structures)
- [Control Flow](#control-flow)
- [Functions and Modules](#functions-and-modules)
- [File Operations](#file-operations)
- [Error Handling](#error-handling)
- [Object-Oriented Programming](#object-oriented-programming)
- [Python for Bioinformatics](#python-for-bioinformatics)
- [Data Analysis Libraries](#data-analysis-libraries)
- [Visualization](#visualization)
- [Bioinformatics Libraries](#bioinformatics-libraries)
- [Advanced Topics](#advanced-topics)
- [Best Practices](#best-practices)
- [Resources and References](#resources-and-references)

## Introduction

Python has become the programming language of choice for bioinformatics due to its readability, versatility, and the rich ecosystem of scientific libraries. This guide will take you from the basics of Python syntax to implementing bioinformatics-specific applications.

## Setting Up Your Environment

### Installing Python
```python
# Python 3.x is recommended for bioinformatics
# Download from python.org or use package managers

# Check your Python version
python --version
```

### Setting Up a Virtual Environment
```python
# Create a virtual environment
python -m venv bioinfo_env

# Activate virtual environment
# On Windows
bioinfo_env\Scripts\activate
# On macOS/Linux
source bioinfo_env/bin/activate

# Install packages
pip install numpy pandas matplotlib biopython
```

### Jupyter Notebooks
```python
# Install Jupyter
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook
```

## Python Basics

### Variables and Assignments
```python
# Variable assignment
dna_sequence = "ATGCTAGCTAGCTAGC"
gene_count = 42
mutation_rate = 0.001

# Multiple assignment
a, b, c = 1, 2, 3

# Check variable type
print(type(dna_sequence))  # <class 'str'>
```

### Basic Operations
```python
# Arithmetic operations
a = 10
b = 3

addition = a + b        # 13
subtraction = a - b     # 7
multiplication = a * b  # 30
division = a / b        # 3.3333...
floor_division = a // b # 3
modulus = a % b         # 1
exponent = a ** b       # 1000

# String operations
seq1 = "ATGC"
seq2 = "GCAT"
combined = seq1 + seq2  # "ATGCGCAT"
repeated = seq1 * 3     # "ATGCATGCATGC"
```

### Input and Output
```python
# Output to console
print("DNA Sequence:", dna_sequence)

# Formatted output
gene_name = "BRCA1"
print(f"Gene {gene_name} has {len(dna_sequence)} base pairs")

# Input from user
user_input = input("Enter a DNA sequence: ")
```

## Data Types and Structures

### Numbers
```python
# Integers
chromosome_count = 23

# Floating point
p_value = 0.05
expression_level = 2.45e-10  # Scientific notation

# Complex numbers (useful for signal processing)
complex_num = 3 + 4j
```

### Strings (Text Sequences)
```python
# DNA sequence as string
dna = "ATGCTAGCTAGCTAGCTGACT"

# String slicing
first_codon = dna[0:3]  # "ATG"
last_codon = dna[-3:]   # "ACT"

# String methods
print(dna.count("ATG"))  # Count occurrences
print(dna.find("TAG"))   # Find position
print(dna.replace("T", "U"))  # DNA to RNA

# Nucleotide frequency
a_count = dna.count("A")
t_count = dna.count("T")
g_count = dna.count("G")
c_count = dna.count("C")
gc_content = (g_count + c_count) / len(dna)
```

### Lists
```python
# List of gene names
genes = ["BRCA1", "TP53", "EGFR", "KRAS"]

# Accessing elements
first_gene = genes[0]  # "BRCA1"
last_gene = genes[-1]  # "KRAS"

# Modifying lists
genes.append("BRAF")
genes.insert(1, "PTEN")
genes.remove("EGFR")
popped_gene = genes.pop()

# List comprehension
lengths = [len(gene) for gene in genes]
long_genes = [gene for gene in genes if len(gene) > 4]

# Nested lists for matrix-like data
expression_matrix = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]
```

### Tuples
```python
# Tuples are immutable (can't be changed after creation)
gene_location = ("chr17", 43044295, 43125483)

# Unpack tuple
chromosome, start, end = gene_location

# Tuple of DNA and RNA sequences
sequences = ("ATGC", "AUGC")
```

### Dictionaries
```python
# Dictionary: key-value pairs
gene_info = {
    "name": "BRCA1",
    "chromosome": "17",
    "start": 43044295,
    "end": 43125483,
    "strand": "+"
}

# Accessing values
gene_name = gene_info["name"]
location = gene_info.get("chromosome", "unknown")

# Adding/updating entries
gene_info["function"] = "DNA repair"
gene_info["end"] = 43125484

# Dictionary comprehension
codon_table = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T"
}
reverse_codon = {amino: codon for codon, amino in codon_table.items()}
```

### Sets
```python
# Sets: unordered collections of unique elements
ref_nucleotides = {"A", "C", "G", "T"}
rna_nucleotides = {"A", "C", "G", "U"}

# Set operations
common = ref_nucleotides & rna_nucleotides  # intersection
all_nucleotides = ref_nucleotides | rna_nucleotides  # union
difference = ref_nucleotides - rna_nucleotides  # difference
```

## Control Flow

### Conditional Statements
```python
# Simple if statement
gc_content = 0.55
if gc_content > 0.5:
    print("High GC content")
elif gc_content > 0.4:
    print("Medium GC content")
else:
    print("Low GC content")

# Ternary conditional expression
result = "High" if gc_content > 0.5 else "Low"
```

### Loops
```python
# For loop with a list
for gene in genes:
    print(f"Processing gene: {gene}")

# For loop with range
for i in range(10):
    print(i)  # 0 to 9

# For loop with enumerate (getting index and value)
for i, nucleotide in enumerate("ATGC"):
    print(f"Position {i}: {nucleotide}")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control statements
for gene in genes:
    if gene == "TP53":
        continue  # Skip to next iteration
    if gene == "KRAS":
        break  # Exit the loop
    print(gene)
```

## Functions and Modules

### Defining Functions
```python
# Basic function
def calculate_gc_content(sequence):
    g_count = sequence.count("G")
    c_count = sequence.count("C")
    return (g_count + c_count) / len(sequence)

# Function with default parameters
def translate_dna(sequence, start_pos=0):
    codon_table = {"ATG": "M", "AAA": "K", "AAG": "K", /* more codons */}
    protein = ""
    for i in range(start_pos, len(sequence), 3):
        codon = sequence[i:i+3]
        if len(codon) == 3:
            protein += codon_table.get(codon, "X")
    return protein

# Function with multiple returns
def find_motif(sequence, motif):
    positions = []
    pos = sequence.find(motif)
    while pos != -1:
        positions.append(pos)
        pos = sequence.find(motif, pos + 1)
    return positions

# Lambda functions (anonymous functions)
complement = lambda seq: seq.replace("A", "t").replace("T", "a").replace("G", "c").replace("C", "g").upper()
```

### Importing Modules
```python
# Import standard library
import math
import random
import os

# Import specific functions from modules
from math import log2, sqrt
from random import shuffle, sample

# Import with alias
import numpy as np
import pandas as pd

# Import custom module
from my_bio_tools import translate, reverse_complement
```

### Creating Modules
```python
# File: dna_utils.py
def complement(sequence):
    """Return the complementary DNA sequence"""
    comp_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp_dict.get(base, base) for base in sequence)

def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence"""
    return complement(sequence)[::-1]

# In another script:
import dna_utils
rc_seq = dna_utils.reverse_complement("ATGC")
```

## File Operations

### Reading and Writing Text Files
```python
# Reading a FASTA file
def read_fasta(filename):
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences

# Writing a FASTA file
def write_fasta(sequences, filename):
    with open(filename, 'w') as file:
        for seq_id, sequence in sequences.items():
            file.write(f">{seq_id}\n")
            # Write sequence in lines of 60 characters
            for i in range(0, len(sequence), 60):
                file.write(f"{sequence[i:i+60]}\n")
```

### Working with CSV Files
```python
# Reading CSV files
import csv

def read_expression_data(filename):
    expression_data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            expression_data.append(row)
    return expression_data

# Writing CSV files
def write_results(results, filename):
    with open(filename, 'w', newline='') as file:
        fieldnames = ['gene', 'p_value', 'fold_change']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
```

### Working with Binary Files
```python
# Reading and writing binary files
import pickle

# Saving data
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Loading data
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
```

## Error Handling

### Try-Except Blocks
```python
# Basic try-except
try:
    result = calculate_gc_content("ATXG")
except:
    print("Error in sequence")

# Specific exceptions
try:
    with open("data.fa", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
    
# Try-except-else-finally
try:
    sequence = "ATGC"
    gc = calculate_gc_content(sequence)
except ValueError as e:
    print(f"Error: {e}")
else:
    print(f"GC content: {gc}")
finally:
    print("Analysis complete")
```

### Raising Exceptions
```python
def validate_dna(sequence):
    valid_bases = set("ATGC")
    if not set(sequence).issubset(valid_bases):
        raise ValueError("Invalid DNA sequence")
    return True

# Using custom exceptions
class SequenceError(Exception):
    """Exception raised for errors in biological sequences"""
    pass

def check_sequence(seq_type, sequence):
    if seq_type == "DNA" and not set(sequence).issubset(set("ATGC")):
        raise SequenceError("Invalid DNA sequence")
    elif seq_type == "RNA" and not set(sequence).issubset(set("AUGC")):
        raise SequenceError("Invalid RNA sequence")
```

## Object-Oriented Programming

### Classes and Objects
```python
# Define a class
class Gene:
    def __init__(self, name, sequence, chromosome=None):
        self.name = name
        self.sequence = sequence
        self.chromosome = chromosome
    
    def length(self):
        return len(self.sequence)
    
    def gc_content(self):
        g_count = self.sequence.count("G")
        c_count = self.sequence.count("C")
        return (g_count + c_count) / self.length()
    
    def __str__(self):
        return f"Gene: {self.name}, Length: {self.length()}"

# Using the class
my_gene = Gene("BRCA1", "ATGCTAGCTAGC", "17")
print(my_gene.length())
print(my_gene.gc_content())
print(my_gene)
```

### Inheritance
```python
# Parent class
class Sequence:
    def __init__(self, sequence):
        self.sequence = sequence
    
    def length(self):
        return len(self.sequence)
    
    def gc_content(self):
        g_count = self.sequence.count("G") + self.sequence.count("g")
        c_count = self.sequence.count("C") + self.sequence.count("c")
        return (g_count + c_count) / self.length()

# Child class
class DNASequence(Sequence):
    def __init__(self, sequence, is_coding=False):
        super().__init__(sequence)
        self.is_coding = is_coding
    
    def complement(self):
        comp_dict = {"A": "T", "T": "A", "G": "C", "C": "G",
                    "a": "t", "t": "a", "g": "c", "c": "g"}
        return "".join(comp_dict.get(base, base) for base in self.sequence)
    
    def reverse_complement(self):
        return self.complement()[::-1]
    
    def transcribe(self):
        return self.sequence.replace("T", "U").replace("t", "u")

# Using inheritance
dna = DNASequence("ATGCTAGCTAGCTGACT")
print(dna.length())
print(dna.gc_content())
print(dna.reverse_complement())
```

## Python for Bioinformatics

### DNA Manipulation
```python
# Reverse complement
def reverse_complement(sequence):
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement.get(base, base) for base in reversed(sequence))

# Transcription (DNA to RNA)
def transcribe(dna_sequence):
    return dna_sequence.replace("T", "U")

# Translation (RNA to protein)
def translate(rna_sequence):
    codon_table = {
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        # ... more codons
    }
    
    protein = ""
    for i in range(0, len(rna_sequence), 3):
        codon = rna_sequence[i:i+3]
        if len(codon) < 3:
            break
        amino_acid = codon_table.get(codon, "X")
        if amino_acid == "Stop":
            break
        protein += amino_acid
    
    return protein
```

### Sequence Analysis
```python
# Find open reading frames (ORFs)
def find_orfs(dna_sequence, min_length=30):
    orfs = []
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]
    
    # Search in all three reading frames
    for frame in range(3):
        for i in range(frame, len(dna_sequence), 3):
            # If we find a start codon
            if dna_sequence[i:i+3] == start_codon:
                # Look for the next in-frame stop codon
                for j in range(i+3, len(dna_sequence), 3):
                    if dna_sequence[j:j+3] in stop_codons:
                        orf_length = j - i + 3
                        if orf_length >= min_length:
                            orfs.append((i, j+3, dna_sequence[i:j+3]))
                        break
    
    return orfs

# Calculate edit distance (Levenshtein distance)
def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    # Create a matrix to store the distances
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Initialize the matrix
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1,  # substitution
                               dp[i-1][j] + 1,     # deletion
                               dp[i][j-1] + 1)     # insertion
    
    return dp[m][n]
```

### Basic Alignment
```python
# Simple global alignment (Needleman-Wunsch)
def global_alignment(seq1, seq2, match=1, mismatch=-1, gap=-1):
    # Initialize the scoring matrix
    m, n = len(seq1), len(seq2)
    score_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Initialize first row and column with gap penalties
    for i in range(m+1):
        score_matrix[i][0] = i * gap
    for j in range(n+1):
        score_matrix[0][j] = j * gap
    
    # Fill the scoring matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate scores for the three possible moves
            match_score = score_matrix[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            delete_score = score_matrix[i-1][j] + gap
            insert_score = score_matrix[i][j-1] + gap
            
            # Take the maximum score
            score_matrix[i][j] = max(match_score, delete_score, insert_score)
    
    # Traceback to find the alignment
    alignment1, alignment2 = "", ""
    i, j = m, n
    
    while i > 0 and j > 0:
        score = score_matrix[i][j]
        score_diag = score_matrix[i-1][j-1]
        score_up = score_matrix[i-1][j]
        score_left = score_matrix[i][j-1]
        
        if score == score_diag + (match if seq1[i-1] == seq2[j-1] else mismatch):
            alignment1 = seq1[i-1] + alignment1
            alignment2 = seq2[j-1] + alignment2
            i -= 1
            j -= 1
        elif score == score_up + gap:
            alignment1 = seq1[i-1] + alignment1
            alignment2 = "-" + alignment2
            i -= 1
        elif score == score_left + gap:
            alignment1 = "-" + alignment1
            alignment2 = seq2[j-1] + alignment2
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        alignment1 = seq1[i-1] + alignment1
        alignment2 = "-" + alignment2
        i -= 1
    while j > 0:
        alignment1 = "-" + alignment1
        alignment2 = seq2[j-1] + alignment2
        j -= 1
    
    return alignment1, alignment2, score_matrix[m][n]
```

## Data Analysis Libraries

### NumPy
```python
import numpy as np

# Create arrays
sequence_lengths = np.array([15, 23, 42, 31, 18])
gene_expression = np.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])

# Array operations
mean_length = np.mean(sequence_lengths)
max_expression = np.max(gene_expression)
total_length = np.sum(sequence_lengths)

# Array manipulation
normalized = (gene_expression - np.mean(gene_expression)) / np.std(gene_expression)
log_expression = np.log2(gene_expression)

# Linear algebra
covariance = np.cov(gene_expression)
correlation = np.corrcoef(gene_expression)
eigenvalues, eigenvectors = np.linalg.eig(covariance)
```

### Pandas
```python
import pandas as pd

# Create DataFrame from dictionary
expression_data = pd.DataFrame({
    'gene': ['BRCA1', 'TP53', 'EGFR', 'KRAS', 'PTEN'],
    'expression': [1.2, 2.3, 0.8, 1.5, 1.1],
    'p_value': [0.01, 0.005, 0.03, 0.02, 0.01]
})

# Read from file
# expression_data = pd.read_csv('expression.csv')

# Basic DataFrame operations
print(expression_data.head())
print(expression_data.describe())
print(expression_data['gene'].value_counts())

# Filtering data
significant = expression_data[expression_data['p_value'] < 0.02]
high_expression = expression_data[expression_data['expression'] > 1.0]
combined_filter = expression_data[(expression_data['p_value'] < 0.02) & 
                                 (expression_data['expression'] > 1.0)]

# Sorting
sorted_by_expr = expression_data.sort_values('expression', ascending=False)

# Grouping and aggregation
grouped = expression_data.groupby('p_value').mean()

# Apply functions to data
expression_data['log_expr'] = expression_data['expression'].apply(np.log2)

# Merge DataFrames
metadata = pd.DataFrame({
    'gene': ['BRCA1', 'TP53', 'EGFR', 'KRAS', 'PTEN'],
    'chromosome': ['17', '17', '7', '12', '10']
})
merged = pd.merge(expression_data, metadata, on='gene')
```

### SciPy
```python
from scipy import stats
import scipy.cluster.hierarchy as sch

# Statistical tests
t_stat, p_value = stats.ttest_ind(control_group, treatment_group)
chi2, p_value = stats.chisquare(observed_frequencies)
r, p_value = stats.pearsonr(gene1_expression, gene2_expression)

# Clustering
Z = sch.linkage(gene_expression, method='ward')
clusters = sch.fcluster(Z, 2, criterion='maxclust')

# Optimization
from scipy import optimize
def objective_function(params):
    # Model fitting function
    return error_metric

result = optimize.minimize(objective_function, initial_params)
```

## Visualization

### Matplotlib
```python
import matplotlib.pyplot as plt

# Basic plotting
plt.figure(figsize=(10, 6))
plt.plot(time_points, gene_expression, marker='o', linestyle='-', color='blue')
plt.title('Gene Expression Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Expression Level')
plt.grid(True)
plt.savefig('expression_plot.png', dpi=300)
plt.show()

# Multiple plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x1, y1)
plt.title('Gene A')

plt.subplot(2, 2, 2)
plt.plot(x2, y2)
plt.title('Gene B')

plt.subplot(2, 2, 3)
plt.scatter(x3, y3)
plt.title('Correlation')

plt.subplot(2, 2, 4)
plt.bar(categories, values)
plt.title('Expression by Tissue')

plt.tight_layout()
plt.show()
```

### Seaborn
```python
import seaborn as sns

# Set theme
sns.set_theme(style="whitegrid")

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = np.corrcoef(expression_matrix.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            xticklabels=gene_names, yticklabels=gene_names)
plt.title('Gene Expression Correlation')
plt.show()

# Distribution plots
plt.figure(figsize=(12, 6))
sns.histplot(expression_data['expression'], kde=True)
plt.title('Distribution of Expression Values')
plt.show()

# Box plots
plt.figure(figsize=(12, 6))
sns.boxplot(x='tissue', y='expression', data=tissue_expression)
plt.title('Expression by Tissue Type')
plt.show()

# Pair plots
sns.pairplot(expression_data, hue='condition')
plt.show()
```

## Bioinformatics Libraries

### Biopython
```python
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Entrez, AlignIO, Phylo
from Bio.Blast import NCBIWWW, NCBIXML

# Set your email for Entrez
Entrez.email = "your.email@example.com"

# Reading sequences
def read_sequences(filename, format="fasta"):
    """Read sequences from a file."""
    sequences = []
    for record in SeqIO.parse(filename, format):
        sequences.append(record)
    return sequences

# Sequence manipulation
dna_seq = Seq("ATGCTAGCTAGCT")
protein = dna_seq.translate()
reverse_complement = dna_seq.reverse_complement()

# Create a sequence record
record = SeqRecord(
    Seq("ATGCTAGCTGATCGATCG"),
    id="example",
    name="example gene",
    description="example gene sequence"
)

# Save sequences
SeqIO.write([record], "output.fasta", "fasta")

# Search NCBI databases
handle = Entrez.efetch(db="nucleotide", id="NM_001126.3", rettype="gb", retmode="text")
record = SeqIO.read(handle, "genbank")
handle.close()

# BLAST search
result_handle = NCBIWWW.qblast("blastn", "nt", dna_seq)
blast_record = NCBIXML.read(result_handle)
for alignment in blast_record.alignments:
    for hsp in alignment.hsps:
        if hsp.expect < 1e-10:
            print(f"Sequence: {alignment.title}")
            print(f"E-value: {hsp.expect}")
```

### scikit-bio
```python
import skbio
from skbio import DNA, RNA, Protein
from skbio.alignment import global_pairwise_align_nucleotide

# Create a DNA sequence
dna = DNA("ATGCTAGCTGATCG")

# Manipulate the sequence
rna = dna.transcribe()
protein = dna.translate()

# Calculate GC content
gc_content = skbio.io.util.GC_calculator(str(dna))

# Alignment
seq1 = DNA("ATGCTAGCTGATCG")
seq2 = DNA("ATGCTAGCTAATCG")
alignment, score, _ = global_pairwise_align_nucleotide(seq1, seq2)
print(alignment)
print(f"Alignment score: {score}")
```

### scikit-learn
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare expression data
# X: expression data (genes x samples)
# y: labels (e.g., disease vs. control)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Classification
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
feature_importance = classifier.feature_importances_
```


## Advanced Topics

### Parallel Processing
```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_sequence(sequence):
    # Some computationally intensive operation
    results = []
    # Analyze sequence, calculate metrics, etc.
    return results

# Using multiprocessing directly
def parallel_process_sequences(sequences):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(process_sequence, sequences)
    pool.close()
    pool.join()
    return results

# Using ProcessPoolExecutor (cleaner API)
def parallel_process_sequences_executor(sequences):
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(process_sequence, sequences):
            results.append(result)
    return results
```

### Generators and Memory-Efficient Processing
```python
# Generator function for reading large FASTA files
def read_fasta_generator(filename):
    """Yield records from FASTA file without loading entire file in memory."""
    current_id = None
    current_seq = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    # Yield the previous sequence before starting a new one
                    yield (current_id, ''.join(current_seq))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget to yield the last sequence
        if current_id:
            yield (current_id, ''.join(current_seq))

# Using the generator
for seq_id, sequence in read_fasta_generator("large_genome.fa"):
    # Process sequences one at a time without loading all into memory
    gc_content = calculate_gc_content(sequence)
    print(f"{seq_id}: {gc_content}")
```

### Sequence Alignment Algorithms
```python
# Local alignment (Smith-Waterman)
def local_alignment(seq1, seq2, match=2, mismatch=-1, gap=-1):
    # Initialize the scoring matrix
    m, n = len(seq1), len(seq2)
    score_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    max_score = 0
    max_pos = (0, 0)
    
    # Fill the scoring matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate scores for the three possible moves
            match_score = score_matrix[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            delete_score = score_matrix[i-1][j] + gap
            insert_score = score_matrix[i][j-1] + gap
            
            # Take the maximum score or 0 (local alignment)
            score_matrix[i][j] = max(0, match_score, delete_score, insert_score)
            
            # Keep track of the maximum score for traceback
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)
    
    # Traceback from the maximum score position
    alignment1, alignment2 = "", ""
    i, j = max_pos
    
    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        score = score_matrix[i][j]
        score_diag = score_matrix[i-1][j-1]
        score_up = score_matrix[i-1][j]
        score_left = score_matrix[i][j-1]
        
        if score == 0:
            break
        elif score == score_diag + (match if seq1[i-1] == seq2[j-1] else mismatch):
            alignment1 = seq1[i-1] + alignment1
            alignment2 = seq2[j-1] + alignment2
            i -= 1
            j -= 1
        elif score == score_up + gap:
            alignment1 = seq1[i-1] + alignment1
            alignment2 = "-" + alignment2
            i -= 1
        elif score == score_left + gap:
            alignment1 = "-" + alignment1
            alignment2 = seq2[j-1] + alignment2
            j -= 1
    
    return alignment1, alignment2, max_score
```

### Hidden Markov Models
```python
# Simple implementation of Viterbi algorithm for HMM decoding
def viterbi(observations, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    for state in states:
        V[0][state] = start_p[state] * emit_p[state][observations[0]]
        path[state] = [state]
    
    # Run Viterbi for t > 0
    for t in range(1, len(observations)):
        V.append({})
        new_path = {}
        
        for state_to in states:
            max_prob = 0
            max_state = None
            
            for state_from in states:
                prob = V[t-1][state_from] * trans_p[state_from][state_to] * emit_p[state_to][observations[t]]
                
                if prob > max_prob:
                    max_prob = prob
                    max_state = state_from
            
            V[t][state_to] = max_prob
            new_path[state_to] = path[max_state] + [state_to]
        
        path = new_path
    
    # Find the most likely final state
    max_prob = 0
    max_state = None
    for state in states:
        if V[len(observations)-1][state] > max_prob:
            max_prob = V[len(observations)-1][state]
            max_state = state
    
    return path[max_state], max_prob

# Example usage for gene prediction
observations = "ATGCTAGCTAGCTGACT"  # DNA sequence
states = ["Coding", "NonCoding"]
start_probability = {"Coding": 0.2, "NonCoding": 0.8}
transition_probability = {
    "Coding": {"Coding": 0.9, "NonCoding": 0.1},
    "NonCoding": {"Coding": 0.2, "NonCoding": 0.8}
}
emission_probability = {
    "Coding": {"A": 0.25, "T": 0.20, "G": 0.30, "C": 0.25},
    "NonCoding": {"A": 0.30, "T": 0.30, "G": 0.20, "C": 0.20}
}

path, probability = viterbi(observations, states, start_probability, 
                           transition_probability, emission_probability)
print(f"Most likely state sequence: {path}")
print(f"Probability: {probability}")
```

### Machine Learning for Genomics
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Example: Predicting transcription factor binding sites
def extract_features(sequence):
    """Extract features from a DNA sequence for ML."""
    features = []
    
    # Basic nucleotide frequency
    features.append(sequence.count('A') / len(sequence))
    features.append(sequence.count('T') / len(sequence))
    features.append(sequence.count('G') / len(sequence))
    features.append(sequence.count('C') / len(sequence))
    
    # GC content
    features.append((sequence.count('G') + sequence.count('C')) / len(sequence))
    
    # Dinucleotide frequencies
    for di in ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
               'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']:
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i:i+2] == di:
                count += 1
        features.append(count / (len(sequence) - 1))
    
    return features

# Prepare training data
X = []  # Features
y = []  # Labels (1 for binding, 0 for non-binding)

# For each sequence in dataset:
for sequence, label in dataset:
    X.append(extract_features(sequence))
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Feature importance
importances = model.feature_importances_
feature_names = ["A_freq", "T_freq", "G_freq", "C_freq", "GC_content"] + \
                ["Di_" + di for di in ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                                       'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']]

# Sort by importance
sorted_indices = np.argsort(importances)[::-1]
print("Feature importance:")
for i in sorted_indices[:10]:  # Top 10 features
    print(f"{feature_names[i]}: {importances[i]:.4f}")
```

### Deep Learning for Bioinformatics
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# One-hot encode DNA sequences
def one_hot_encode(sequences, max_length=100):
    # Define encoding dictionary
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
                'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 
                'N': [0, 0, 0, 0]}
    
    # Initialize array
    X = np.zeros((len(sequences), max_length, 4), dtype=np.float32)
    
    # Encode sequences
    for i, seq in enumerate(sequences):
        seq = seq[:max_length].upper()  # Truncate if longer than max_length
        for j, base in enumerate(seq):
            X[i, j] = encoding.get(base, encoding['N'])
    
    return X

# Build a CNN model for sequence classification
def build_cnn_model(seq_length=100, num_classes=2):
    model = Sequential([
        # Convolutional layers
        Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(seq_length, 4)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=8, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=256, kernel_size=8, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example usage
sequences = ["ATGCTAGCTAGCT", "GCTAGCTAGCTAG", ...] # Your DNA sequences
labels = [0, 1, ...]  # Your labels

# Preprocess
X = one_hot_encode(sequences)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
model = build_cnn_model(seq_length=X.shape[1], num_classes=len(np.unique(y)))

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

### Handling Next-Generation Sequencing Data
```python
# Using pysam to work with BAM/SAM files
import pysam

def analyze_bam_file(bam_file, reference_genome):
    # Open BAM file for reading
    bamfile = pysam.AlignmentFile(bam_file, "rb")
    
    # Get reference sequence
    reference = pysam.FastaFile(reference_genome)
    
    # Iterate through alignments
    coverage = {}
    variants = []
    
    for read in bamfile.fetch():
        # Skip unmapped reads
        if read.is_unmapped:
            continue
        
        # Get reference name (chromosome)
        ref_name = read.reference_name
        
        # Track coverage
        if ref_name not in coverage:
            coverage[ref_name] = {}
        
        for pos in read.get_reference_positions():
            if pos not in coverage[ref_name]:
                coverage[ref_name][pos] = 0
            coverage[ref_name][pos] += 1
        
        # Simple variant calling (just for demonstration)
        for read_pos, ref_pos, ref_base in read.get_aligned_pairs(with_seq=True):
            if read_pos is None or ref_pos is None:
                continue  # Skip insertions/deletions
            
            read_base = read.query_sequence[read_pos].upper()
            ref_base = ref_base.upper()
            
            if read_base != ref_base and ref_base != 'N':
                variants.append({
                    'chrom': ref_name,
                    'position': ref_pos,
                    'ref': ref_base,
                    'alt': read_base
                })
    
    # Close files
    bamfile.close()
    reference.close()
    
    return coverage, variants

# Handling FASTQ files
def process_fastq(fastq_file, quality_threshold=20):
    """Process FASTQ file and filter by quality."""
    high_quality_seqs = []
    
    with open(fastq_file, 'r') as file:
        while True:
            # Read four lines at a time (FASTQ format)
            header = file.readline().strip()
            if not header:
                break  # End of file
            
            sequence = file.readline().strip()
            plus_line = file.readline().strip()
            quality = file.readline().strip()
            
            # Filter by quality
            if average_quality(quality) >= quality_threshold:
                high_quality_seqs.append((header, sequence, quality))
    
    return high_quality_seqs

def average_quality(quality_string):
    """Calculate average quality from FASTQ quality string."""
    # Convert ASCII quality characters to Phred+33 scores
    quality_scores = [ord(char) - 33 for char in quality_string]
    return sum(quality_scores) / len(quality_scores) if quality_scores else 0
```

### Molecular Phylogenetics
```python
# Using BioPython for phylogenetic analysis
from Bio import Phylo, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

def build_phylogenetic_tree(alignment_file, format="fasta", method="nj"):
    """Build a phylogenetic tree from a sequence alignment."""
    # Read the alignment
    alignment = AlignIO.read(alignment_file, format)
    
    # Calculate distance matrix
    calculator = DistanceCalculator('identity')
    distance_matrix = calculator.get_distance(alignment)
    
    # Build the tree
    constructor = DistanceTreeConstructor()
    if method == "nj":
        tree = constructor.nj(distance_matrix)  # Neighbor-Joining method
    elif method == "upgma":
        tree = constructor.upgma(distance_matrix)  # UPGMA method
    else:
        raise ValueError("Method must be 'nj' or 'upgma'")
    
    return tree

# Using the function
tree = build_phylogenetic_tree("sequences.fasta", method="nj")

# Visualize the tree (requires matplotlib)
Phylo.draw(tree)

# Save the tree in Newick format
Phylo.write(tree, "phylogenetic_tree.newick", "newick")
```

### Network Analysis in Bioinformatics
```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create protein-protein interaction network
def build_ppi_network(interaction_data):
    """Build protein-protein interaction network from data."""
    G = nx.Graph()
    
    # Add interactions as edges
    for interaction in interaction_data:
        protein1, protein2, confidence = interaction
        G.add_edge(protein1, protein2, weight=confidence)
    
    return G

# Analyze network
def analyze_network(G):
    """Analyze properties of biological network."""
    results = {}
    
    # Basic network statistics
    results['num_nodes'] = G.number_of_nodes()
    results['num_edges'] = G.number_of_edges()
    
    # Degree centrality (number of connections)
    degree_centrality = nx.degree_centrality(G)
    results['highest_degree_centrality'] = max(degree_centrality.items(), key=lambda x: x[1])
    
    # Betweenness centrality (nodes that bridge communities)
    betweenness_centrality = nx.betweenness_centrality(G)
    results['highest_betweenness'] = max(betweenness_centrality.items(), key=lambda x: x[1])
    
    # Clustering coefficient (how connected a node's neighbors are)
    results['avg_clustering'] = nx.average_clustering(G)
    
    # Connected components
    connected_components = list(nx.connected_components(G))
    results['num_connected_components'] = len(connected_components)
    results['largest_component_size'] = len(max(connected_components, key=len))
    
    return results

# Visualize network
def visualize_network(G, node_color_attribute=None, filename=None):
    """Visualize biological network."""
    plt.figure(figsize=(12, 12))
    
    # Node positions using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Node colors based on attribute
    if node_color_attribute:
        node_colors = [G.nodes[node].get(node_color_attribute, 0) for node in G.nodes()]
    else:
        node_colors = 'skyblue'
    
    # Node sizes based on degree
    node_sizes = [300 * nx.degree_centrality(G)[node] + 50 for node in G.nodes()]
    
    # Edge widths based on weight
    edge_widths = [G[u][v].get('weight', 1) for u, v in G.edges()]
    
    # Draw the network
    nx.draw_networkx(
        G, pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        alpha=0.7,
        with_labels=True,
        font_size=8
    )
    
    plt.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
```

## Best Practices

### Code Organization
```python
# Project structure example
"""
bioinformatics_project/
│
├── data/                  # Raw and processed data
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── notebooks/             # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_visualization.ipynb
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── data/              # Data processing scripts
│   │   └── __init__.py
│   ├── features/          # Feature engineering
│   │   └── __init__.py
│   ├── models/            # Analysis and modeling
│   │   └── __init__.py
│   └── visualization/     # Plotting and visualization
│       └── __init__.py
│
├── tests/                 # Unit tests
│   └── test_*.py
│
├── environment.yml        # Conda environment file
├── setup.py               # Package installation
├── README.md              # Project documentation
└── .gitignore             # Files to ignore in git
"""
```

### Documentation
```python
def translate_dna(sequence, start_pos=0, table=1):
    """
    Translate a DNA sequence to protein using the specified genetic code.
    
    Parameters
    ----------
    sequence : str
        DNA sequence to translate
    start_pos : int, optional
        Position to start translation (default: 0)
    table : int, optional
        NCBI genetic code table (default: 1, standard code)
        
    Returns
    -------
    str
        Translated protein sequence
        
    Examples
    --------
    >>> translate_dna("ATGTTGCAG")
    'MLQ'
    
    >>> translate_dna("ATGTTGCAG", table=2)
    'ML*'
    
    Notes
    -----
    This function translates DNA directly without checking for valid
    start codons or reading frames.
    """
    # Function implementation...
```

### Logging
```python
import logging

# Set up logging
def setup_logger(log_file=None, level=logging.INFO):
    """Set up logger for tracking program execution."""
    logger = logging.getLogger('bioinformatics')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Example usage
logger = setup_logger('pipeline.log')
logger.info("Starting analysis pipeline")
logger.debug("Loading sequence data from %s", filename)

try:
    # Some code that might fail
    result = process_sequences(sequences)
    logger.info("Successfully processed %d sequences", len(sequences))
except Exception as e:
    logger.error("Failed to process sequences: %s", str(e))
    logger.exception("Full traceback:")
```

### Testing
```python
# File: test_dna_utils.py
import unittest
from src.utils import dna_utils

class TestDNAUtils(unittest.TestCase):
    
    def test_complement(self):
        """Test DNA complement function."""
        self.assertEqual(dna_utils.complement("ATGC"), "TACG")
        self.assertEqual(dna_utils.complement(""), "")
        self.assertEqual(dna_utils.complement("N"), "N")
    
    def test_reverse_complement(self):
        """Test reverse complement function."""
        self.assertEqual(dna_utils.reverse_complement("ATGC"), "GCAT")
        self.assertEqual(dna_utils.reverse_complement(""), "")
    
    def test_gc_content(self):
        """Test GC content calculation."""
        self.assertEqual(dna_utils.calculate_gc_content("ATGC"), 0.5)
        self.assertEqual(dna_utils.calculate_gc_content("AAAA"), 0.0)
        self.assertEqual(dna_utils.calculate_gc_content("GGCC"), 1.0)
        with self.assertRaises(ValueError):
            dna_utils.calculate_gc_content("")

if __name__ == '__main__':
    unittest.main()
```

### Virtual Environments and Reproducibility
```python
# Creating a conda environment file (environment.yml)
"""
name: bioinfo-env
channels:
  - bioconda
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - numpy=1.20
  - pandas=1.3
  - matplotlib=3.4
  - seaborn=0.11
  - biopython=1.79
  - scikit-learn=0.24
  - tensorflow=2.6
  - networkx=2.6
  - jupyter=1.0
  - pytest=6.2
  - pip=21.2
  - pip:
    - pysam==0.18
    - scikit-bio==0.5.6
"""

# Using pip requirements.txt
"""
numpy==1.20.3
pandas==1.3.4
matplotlib==3.4.3
seaborn==0.11.2
biopython==1.79
scikit-learn==0.24.2
tensorflow==2.6.0
networkx==2.6.3
jupyter==1.0.0
pytest==6.2.5
pysam==0.18.0
scikit-bio==0.5.6
"""

# Create and activate environment
# With conda:
# conda env create -f environment.yml
# conda activate bioinfo-env

# With pip and venv:
# python -m venv bioinfo-env
# source bioinfo-env/bin/activate  # On Linux/Mac
# bioinfo-env\Scripts\activate     # On Windows
# pip install -r requirements.txt
```

### Performance Optimization
```python
# Profiling code execution
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile a function's execution."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)  # Print top 20 time-consuming functions
    
    return result

# Memory usage monitoring
import tracemalloc

def monitor_memory(func, *args, **kwargs):
    """Monitor memory usage of a function."""
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    
    return result

# Vectorizing operations
import numpy as np

# Slow way (using Python loops)
def calculate_gc_contents_slow(sequences):
    results = []
    for seq in sequences:
        g_count = seq.count('G')
        c_count = seq.count('C')
        gc_content = (g_count + c_count) / len(seq)
        results.append(gc_content)
    return results

# Fast way (using NumPy vectorization)
def calculate_gc_contents_fast(sequences):
    # Convert to NumPy array of characters
    seq_array = np.array([list(seq) for seq in sequences])
    
    # Vectorized counting
    g_mask = (seq_array == 'G')
    c_mask = (seq_array == 'C')
    
    # Count G's and C's for each sequence
    g_counts = g_mask.sum(axis=1)
    c_counts = c_mask.sum(axis=1)
    
    # Calculate GC content
    sequence_lengths = np.array([len(seq) for seq in sequences])
    gc_contents = (g_counts + c_counts) / sequence_lengths
    
    return gc_contents
```

## Resources and References

### Online Resources
- Python Documentation: https://docs.python.org/3/
- BioPython Documentation: https://biopython.org/wiki/Documentation
- SciPy Ecosystem: https://scipy.org/
- Pandas Documentation: https://pandas.pydata.org/docs/
- NumPy Documentation: https://numpy.org/doc/stable/
- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html

### Bioinformatics-Specific Resources
- NCBI Resources: https://www.ncbi.nlm.nih.gov/
- Ensembl Genome Browser: https://www.ensembl.org/
- UCSC Genome Browser: https://genome.ucsc.edu/
- Galaxy Project: https://usegalaxy.org/
- European Bioinformatics Institute: https://www.ebi.ac.uk/
