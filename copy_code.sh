#!/bin/bash

arquivos_python=(`find . -iname "*.py"`)

echo ">>>>>> Estrutura do projeto minhas economias <<<<<<"
echo ""
tree
echo ""


for i in "${arquivos_python[@]}"; do
    echo ">>>>>> Arquivos em ANSIBLE <<<<<<"
    echo "arquivo: $i"; echo ""; echo '```'; echo ""; cat "$i"; echo ""; echo '```'; echo ""
done

