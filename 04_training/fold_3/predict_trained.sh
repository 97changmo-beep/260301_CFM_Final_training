#!/bin/sh
cd /cfmid/public/predict
grep -v "^#" test_compounds.txt | grep -v "^$" | while IFS="	" read -r CID SMILES PEPMASS; do
    echo "$CID	$SMILES"
done | xargs -P 32 -I {} sh -c '
    CID=$(echo "{}" | cut -f1)
    SMILES=$(echo "{}" | cut -f2)
    cfm-predict "$SMILES" 0.001 param_output.log config.txt 0 "predictions/${CID}.log" 2>/dev/null
    echo "  trained: $CID"
'
