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
python pMTnet.py -input input.csv -library library -output output_dir -output_log test/output/output.log
```
* input.csv: input csv file with 3 columns named as "CDR3,Antigen,HLA": TCR-beta CDR3 sequence, peptide sequence, and HLA allele.\
![Input_file_example](https://github.com/tianshilu/pMTnet/blob/master/example_pic/input_file_example.png)
* library: diretory to the downloaded library
* output_dir : diretory you want to save the output
* output_log : local directory to log file with CDR, Antigen, HLA information and predicted binding rank.\
![Output file example](https://github.com/tianshilu/pMTnet/blob/master/example_pic/output_file_example.png)

## Example 
The example input file is under test/input/.
Comand :
```
python pMTnet.py -input test/input/test_input.csv -library library -output test/output -output_log test/output/output.log
```
The output for test_input.csv is under test/output.
