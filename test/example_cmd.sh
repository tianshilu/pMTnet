#!bin/bash
source activate ternary_binding
python pMTnet.py -input test/input/test_input.csv -library library -output test/output -output_log test/output/output.log
