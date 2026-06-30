#!/bin/bash

#Code to run all fields for a provided VOI ID, moving files to and from necessary locations. Based on realRA_v2021


script_param = ( $@ )

centreID = VC
ptID = ${script_param[0]}
plan = 02
Nbatch = 80

unset script_param[0]
script_param = ( ${script_param[@]} ) 

echo $pwd
