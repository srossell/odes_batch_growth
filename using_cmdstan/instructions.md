# Using cmd stan.

## Installing cmdstan
git clone https://github.com/stan-dev/cmdstan.git --recursive
make build  # build tools (optional)


## Write a stan model string
**write_stan_model2txt.py**

Writes the model to a simple text file

## Write data as a R object
**write_data4cmdstan.py**

Creates an R object form a python dictionary using pystan.misc.stan_rdump

## Compile the stan model
**compile_batch_growth**

This is a bashscript that changes to the directory where cmdstan was installed,
then runs make with the file: batch_growth.stan, as target. Note that in the
script you make batch_growth (without ".stan")

To run this script type:
source compile_batch_growth

## Run the compiled model
**run4chains_batch_growth**

source run4chains_batch_growth

## NEXT use arviz to read in the posterior files
pip install arviz
...

