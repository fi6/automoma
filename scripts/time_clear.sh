docker run -it --rm -v /:/host ubuntu:22.04 bash

chmod -R 777 /host/home/xinhai/projects/automoma/output

python scripts/convert_automoma_to_dp3.py --mode collect
python scripts/convert_automoma_to_dp3.py --mode convert
python scripts/convert_automoma_to_dp3.py --mode check
python scripts/convert_automoma_to_dp3.py --mode clean

rsync -avP ~/projects/automoma/baseline/RoboTwin/policy/DP3/data_compressed xinhai@192.168.31.227:~/projects/automoma/baseline/RoboTwin/policy/DP3/data_compressed