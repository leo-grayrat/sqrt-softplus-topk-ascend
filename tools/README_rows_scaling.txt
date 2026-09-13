Run on the CANN machine:

cd ~/sqrt-softplus-topk-ascend
git switch profile-topk-cost-model
git pull --ff-only
source /home/leo/Ascend/cann-9.0.0/set_env.sh
python3 tools/check_rows_scaling_harness.py
python3 tools/check_rows_scaling_scripts.py
bash -n tools/profile_rows_scaling.sh
python3 -m py_compile tools/check_rows_scaling_harness.py tools/check_rows_scaling_scripts.py tools/summarize_rows_scaling.py
bash tools/profile_rows_scaling.sh Ascend920A 1024 8 dav-2201
python3 tools/summarize_rows_scaling.py profile-output/Ascend920A_rowscaling_e1024_k8
