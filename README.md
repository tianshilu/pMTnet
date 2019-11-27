# pMTnet
Deep learning neural network prediction tcr binding specificity to peptide and HLA. 

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
* input.csv: input csv file with 3 columns named as "CDR3,Antigen,HLA": TCR-beta CDR3 sequence, peptide sequence, and HLA allele.\
![Input_file_example](https://github.com/tianshilu/pMTnet/blob/master/example_pic/input_file_example.png)
* model : local directory to h5_file
* embeding_vector_tcr: local directory to Atchley_factor.csv
* hla_db : local directory to hla_library folder
* output : local directory of encoding output
* output_log : local directory to log file
* tcr_encoding_dim : input length of TCR
* paired : whether encode TCR pmhc together (T) or not (F)

2. Predict TCR Binding Rank
``` 
python prediction_main.py -model weights.h5 -encoding_folder encoding_output_folder -output_dir prediction_output_folder
```
* model : local directory to weights.h5 file
* encoding_folder: local directory of output for the first encoding step
* output_dir : local directory to save prediction ou
