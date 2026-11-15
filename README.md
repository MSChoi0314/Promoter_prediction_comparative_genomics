# Promoter prediction and comparative genomics in the *Bacillus licheniformis*–*B. paralicheniformis* clade

The repository covers three parts of the analysis:

1. **Promoter model** – transcription start sites (TSSs) are identified from dRNA-seq data of *B. licheniformis*, 100 bp promoter sequences are extracted and a convolutional neural network (CNN) is trained to distinguish promoters from CDS sequences.
2. **Genome-wide promoter prediction** – the model is applied to 100 bp windows of *B. licheniformis* and *B. paralicheniformis* genomes and the predictions are filtered with the genome annotation.
3. **Promoter-to-gene mapping** – promoters selected as group-specific are linked to the downstream operon, and the operon proteins are compared across genomes with BLASTP.

## Data

### Input data used in the study

- dRNA-seq data of *B. licheniformis*: NCBI BioProject PRJNA188930 (Wiegand et al., 2013).
- Reference genome: *B. licheniformis* ATCC 14580, accession: GCF_034478925.1.
- *B. licheniformis* and *B. paralicheniformis* genomes: genome metadata are in `data/B.lichen_paralichen_metadata.xlsx`

### 1. Promoter model (`scripts/1_promoter_model`)

**1.1 TSS detection and promoter extraction** – `01_detect_tss.py`
- Prediction of TSS and extraction of promoter based on TSS.

**1.2 Removal of TSSs inside CDS** – `02_filter_tss.py`
- Removes TSSs that fall inside a CDS on the same strand and adds the distances to the upstream and downstream CDS.

**1.3 Negative set** – `03_create_negative_set.py`
- Samples non-overlapping 100 bp windows without `N` from CDS regions, excluding the first 100 bp of each CDS.
- Need to create dataset composed of positive set(promoter) and negative set.

**1.4 CNN training with Optuna** – `04_train_cnn_optuna.py`
- Train the CNN model with the dataset. Creates multiple models with multiple configurations.

**1.5 Evaluation on the held-out test set** – `05_evaluate_test_set.py`
- Reports TP, FP, FN, TN, AUC, accuracy, precision and recall (threshold 0.5) for one model or for every `.keras` file in a folder.

### 2. Genome-wide promoter prediction (`scripts/2_genome_promoter_prediction`)
**2.1 100 bp windows** – `01_genome_windows_forward.py`, `01_genome_windows_reverse.py`
- Each genome is split into 100 bp windows with a 10 bp step, on the forward strand and on the reverse complement. Output files are written into each genome subfolder: `<accession>_sequences.csv` and `<accession>_sequences_reverse.csv`.
- In the reverse files, `start_in_scaffold` is counted on the reverse complement (1 = last base of the scaffold).

**2.2 Prediction** – `02_predict_promoters.py`
- Predicts promoter for every window CSV.

**2.3 Filtering** – `03_filter_predictions.py`
- Filter promoter inside CDS and combine adjacent promoters into single promoter.

**2.4 Promoter FASTA** – `04_promoters_to_fasta.py`
- Writes the kept promoters of each genome to FASTA.

### 3. Genes transcribed from group-specific promoters (`scripts/3_promoter_to_gene`)

**3.1 Promoter → operon proteins** – `01_promoter_to_operon_forward.py`, `01_promoter_to_operon_reverse.py`
- For each promoter, the closest downstream CDS on the same strand within 100 bp of the promoter end is found in `genomic.gff`.
- Operon-mapper needs to be ran prior to this step to get the results.

**3.2 Protein FASTA** – `02_operon_proteins_to_fasta.py`
- Collects the proteins from all CSV files in a folder into one FASTA.

**3.3 BLASTP against every genome** – `03_blastp_best_hits.py`
- For each query protein, records the best hit (highest bit-score) in the `protein.faa` of every genome subfolder, together with the self bit-score of the query.

**3.4 Bit-score normalisation** – `04_normalize_bitscore.py`
- Keeps the bit-score rows and divides each value by the self bit-score.

