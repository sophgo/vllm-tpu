#!/bin/bash

port=8000
# python3 prefix_test.py --questions_file long_question_0.txt --port 8000
# script -f -c "python3 prefix_test.py --questions_file long_question_0.txt --port 8000" run_question0.log
for ((i=1; i<=4; i++))
do
    script -f -c "python3 prefix_test.py --questions_file long_question_$i.txt --port $port" run_question$i.log
done