# # generate rosetta input txt
# python /home/lily/SBDD/PepGLAD/evaluation/dG/cal_rosetta_step1.py \
#   --filtered_filename TSLP_align_pep0_opt_fixed_6959

# run rosetta (--rfdiff_relax default is False)
# python run.py --results /home/lily/SBDD/PepGLAD/evaluation/dG/TSLP_align_pep0_opt_fixed_6959.txt 
python run.py --results /home/yangziqing/PocketXMol/outputs_test/pepdesign_pepbdb/base_pxm_20260325_130506/pepbdb_rosetta_gt.txt