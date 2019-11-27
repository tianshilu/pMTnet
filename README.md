# pMTnet
Deep Learning the T Cell Receptor Binding Specificity of Neoantigen
## Dependencies
python(version>3.0.0) ; 
tensorflow (version=1.11.0) ; 
numpy (version=1.16.3) ; 
keras (version=2.2.4) ; 
pandas (version=0.23.4) ; 
scikit-learn (version=0.20.3) ; 
scipy (version=1.2.1)
## Guided Tutorial
1. Encoding For TCR, peptide, and HLA.
Command:
```
python encoding/encoding_main.py -file input.csv -model h5_file -embeding_vectors_tcr h5_file/Atchley_factors.csv -hla_db hla_library/ -output encoding_output_folder -output_log output_folder/ternary.log -tcr_encoding_dim 80 -paired T
```
* input.csv: input csv file with 3 columns named as "CDR3,Antigen,HLA": TCR-beta CDR3 sequence, peptide sequence, and HLA allele.
![Input_file_example](https://github.com/tianshilu/pMTnet/blob/master/input_file_example.png)

2. Predict TCR Binding Rank
``` 
python prediction_main.py -model weights.h5 -encoding_folder encoding_output_folder -output_dir prediction_output_folder
```
