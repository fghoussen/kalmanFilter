#!/bin/bash -eu

python3 -m coverage run --concurrency=thread tst.py

python3 -m coverage report -m \
       |                      \
       awk 'BEGIN{cov = 0;}
            {print $0; if ($1 == "TOTAL") {split($4, tokens, "%"); cov=tokens[1];}}
            END{print "cov = " cov; if (strtonum(cov) < 80) {print "KO - cov regression"; exit(1);};}'
