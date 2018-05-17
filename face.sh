#! /bin/bash
# exec 1>$PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS OPTIONS =====
#
#PBS -N "face_vid"
#PBS -q titan
#PBS -l walltime=400:00:00
#PBS -l nodes=1:ppn=1:gpus=2,mem=16GB
#PBS -V
#
# ===== END PBS OPTIONS =====
#
# ==== Main ======
cd $PBS_O_WORKDIR
mkdir log
{
module load deepgaze/0.1-cuda8 face-py-faster-rcnn/1.0-cuda8 tensorflow/1.3.1-cuda8
python face.py
} > log/output_"$PBS_JOBNAME"_$PBS_JOBID 2>log/errorLog_"$PBS_JOBNAME"_$PBS_JOBID
