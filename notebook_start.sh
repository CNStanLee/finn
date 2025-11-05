export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2022.2
source /tools/Xilinx/Vivado/2022.2/settings64.sh
export VIVADO_PATH=/tools/Xilinx/Vivado/2022.2
export JUPYTER_PORT=8886
export NVIDIA_VISIBLE_DEVICES=all


./run-docker.sh notebook


#lsof -i :8081
#kill -9 12345
#kill -9 $(lsof -ti :8081)


# (optional) NVIDIA_VISIBLE_DEVICES (default “”) Possible values are a comma-separated list of GPU UUID(s) or index(es) e.g. 0,1,2, all, none, 