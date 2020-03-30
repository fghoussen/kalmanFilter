#!/bin/bash -eu

pylint --module-naming-style=camelCase          \
       --const-naming-style=camelCase           \
       --class-naming-style=camelCase           \
       --function-naming-style=camelCase        \
       --method-naming-style=camelCase          \
       --attr-naming-style=camelCase            \
       --argument-naming-style=camelCase        \
       --variable-naming-style=camelCase        \
       --class-attribute-naming-style=camelCase \
       --inlinevar-naming-style=camelCase       \
       --extension-pkg-whitelist=PyQt5          \
       --disable=E1136,R0904,C0302              \
       --max-attributes=8                       \
       kalmanFilter.py                          \
       |                                        \
       awk 'BEGIN{rate = 0;}
            {print $0; if ($2 == "code" && $5 == "rated") {split($7, tokens, "/"); rate=tokens[1];}}
            END{print "rate = " rate; if (strtonum(rate) < 9.75) {print "KO - rate regression"; exit(1);};}'
