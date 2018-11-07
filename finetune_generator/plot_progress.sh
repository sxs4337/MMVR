#!/bin/bash

filename="log.txt"
output_file="loss.txt"
plot_file="loss_plot.jpg"

# Clean up
rm ${output_file}

iter=0
recon_loss=0
feat_loss=0
real_loss=0
fake_loss=0
gen_fake_loss=0

while IFS='' read -r line || [[ -n "$line" ]]; do
  
  if [[ "${line}" =~ "Iteration" ]] ; then
    iter=`echo ${line} | cut -d ":" -f 3 | cut -d " " -f 4`
  elif [[ "${line}" =~ "img loss" ]]; then
    recon_loss=`echo ${line} | cut -d "=" -f 2`
  elif [[ "${line}" =~ "feat loss" ]]; then
    feat_loss=`echo ${line} | cut -d "=" -f 2`
  elif [[ "${line}" =~ "D real loss" ]]; then
    real_loss=`echo ${line} | cut -d "=" -f 2`
  elif [[ "${line}" =~ "G loss" ]]; then
    gen_fake_loss=`echo ${line} | cut -d "=" -f 2`
    echo "${iter}${recon_loss}${feat_loss}${real_loss}${fake_loss}${gen_fake_loss}" >> ${output_file}
  elif [[ "${line}" =~ "D fake loss" ]]; then
    fake_loss=`echo ${line} | cut -d "=" -f 2`
  fi

done < "${filename}"

# Plot the file
python ./plot_loss.py

open ${plot_file}
