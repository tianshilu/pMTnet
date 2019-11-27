# pMTnet
Deep Learning the T Cell Receptor Binding Specificity of Neoantigen
## Dependencies

1. Encoding For TCR, peptide, and HLA.
Command:
```
python encoding_main.py -file input.csv -model h5_file -embeding_vectors_tcr h5_file/Atchley_factors.csv -hla_db hla_library/ -output output -output_log output/ternary.log -tcr_encoding_dim 80 -paired T
```
