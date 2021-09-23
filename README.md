# pMTnet
Deep learning neural network prediction tcr binding specificity to peptide and HLA based on peptide sequences. Please refer to our paper for more details: 'Deep learning-based prediction of T cell receptor-antigen binding specificity.'(https://www.nature.com/articles/s42256-021-00383-2) Lu, T., Zhang, Z., Zhu, J. et al. 2021.
![preview](https://github.com/tianshilu/pMTnet/blob/master/example_pic/pic1.png)
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
python pMTnet.py -input input.csv -library library -output output_dir -output_log test/output/output.log
```
* input.csv: input csv file with 3 columns named as "CDR3,Antigen,HLA": TCR-beta CDR3 sequence, peptide sequence, and HLA allele.\
![Input_file_example](https://github.com/tianshilu/pMTnet/blob/master/example_pic/input_file_example.png)
For more details about CDR3 encoding, please refer to https://github.com/jcao89757/TESSA.
* library: diretory to the downloaded library
* output_dir : diretory you want to save the output
* output_log : local directory to log file with CDR, Antigen, HLA information and predicted binding rank.\


## Example 
The example input file is under test/input/.\
Comand :
```
python pMTnet.py -input test/input/test_input.csv -library library -output test/output -output_log test/output/output.log
```
The output for test_input.csv is under test/output.

## Output file example
pMTnet outputs a table with 4 columns: CDR3 sequences, antigens sequences, HLA alleles, and ranks for each pair of TCR/pMHC. The rank reflects the percentile rank of the predicted binding strength between the TCR and the pMHC with respect to the 10,000 randomly sampled TCRs against the same pMHC. A lower rank considered a good prediction. The sequences of 10,000 background TCRs can be fold under https://github.com/tianshilu/pMTnet/tree/master/library/bg_tcr_library. 
![Output file example](https://github.com/tianshilu/pMTnet/blob/master/example_pic/output_file_example.png)
