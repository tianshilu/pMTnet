# pMTnet
Deep learning neural network prediction tcr binding specificity to peptide and HLA. 
![preview](https://github.com/tianshilu/pMTnet/blob/master/example_pic/flow_chart.png)
## Dependencies
python(version>3.0.0) ; 
tensorflow (version>1.5.0) ; 
numpy (version=1.16.3) ; 
keras (version=2.2.4) ; 
pandas (version=0.23.4) ; 
scikit-learn (version=0.20.3) ; 
scipy (version=1.2.1)
## Guided Tutorial
Command:
```

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

