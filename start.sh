#sudo su luban
#source activate python38
#env
#export HOME=/home/luban
export PATH=/home/luban/environment/anaconda2/envs/python38/bin:/home/luban/environment/anaconda2/condabin:/usr/local/nvidia/bin/:/home/luban/environment/anaconda2/bin:/usr/local/hive-current/bin:/usr/local/hadoop-current/bin:/usr/local/spark-current/bin:/usr/local/java-current/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/luban/.local/bin:/home/luban/bin
#python /nfs/project/wangyuandong/Gallat_GPU/train.py
#env
conda list
nohup python -u /nfs/project/wangyuandong/DDWP/Gallat_multi-GPU/train.py > multi-GPU.log 2>&1