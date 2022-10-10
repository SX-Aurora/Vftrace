#!/bin/bash

generate_function_name () {
   echo "${1}_${2}_${3}"
}

generate_function () {
   local func_name=$1
   local l=$2
   local lmax=$3
   local m=$4
   echo "int $(generate_function_name ${func_name} ${l} ${m})(int i) {"
   if [[ ${l} == ${lmax} ]] ; then
      echo "   return 1+i;"
   else
      local lc=$(bc <<< "${l}+1")
      local mc1=$(bc <<< "${m}*2")
      local mc2=$(bc <<< "${mc1}+1")
      echo "   return" \
           "$(generate_function_name ${func_name} ${lc} ${mc1})(i)" \
           "+" \
           "$(generate_function_name ${func_name} ${lc} ${mc1})(i)" \
           ";"
   fi
   echo "}"
}

generate_function_rec () {
   local func_name=$1
   local l=$2
   local lmax=$3
   local m=$4
   if [[ ${l} != ${lmax} ]] ; then
      local lc=$(bc <<< "${l}+1")
      local mc1=$(bc <<< "${m}*2")
      local mc2=$(bc <<< "${mc1}+1")
      generate_function_rec ${func_name} ${lc} ${lmax} ${mc1}
      generate_function_rec ${func_name} ${lc} ${lmax} ${mc2}
   fi
   generate_function ${func_name} ${l} ${lmax} ${m}
   echo ""
}

generate_function_tree () {
   local func_name=$1
   local lmax=$2
   generate_function_rec ${func_name} 0 ${lmax} 0
}

func_name=pfunc
lmax=6
outname=protofuncts
generate_function_tree ${func_name} ${lmax} > ${outname}.c
echo "int $(generate_function_name ${func_name} 0 0)(int i);" > ${outname}.h
echo "#define LMAX ${lmax}" >> ${outname}.h
