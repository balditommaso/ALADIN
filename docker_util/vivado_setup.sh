#! /usr/bin/env bash


export XILINX_VIVADO=/opt/vivado/Vivado
if [ -n "${PATH}" ]; then
  export PATH=/opt/vivado/Vivado/bin:$PATH
else
  export PATH=/opt/vivado/Vivado/bin
fi