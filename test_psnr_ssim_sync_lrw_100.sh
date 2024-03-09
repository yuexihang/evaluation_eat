#name='qvt_img_pca_sync_3_504_505_lrw_norm'
name=deepprompt_eam3d_all_final_313_lrw_norm_wukong0
device=0
start_frame=0
#rm -r "/data4/talking_head_testing/25fps_video/pcavs_crop/${name}_lrw_filter100"

#scp gy@192.168.188.10://data1/gy/ost/${name}.tar.gz   ./

## uncompress
#tar -zxf ${name}.tar.gz

cd './code'
#----------------------------------------------##
# preprocess
python preprocess_lrw.py --save_name ${name}_lrw_filter100 --fake_pth "../result/${name}/*.mp4" --name_mode 1  --ours_filter_100 --need_align_crop
#----------------------------------------------##

#----------------------------------------------##
# ## fast alignment
CUDA_VISIBLE_DEVICES=${device} python _fast_align_lrw.py --name ${name}_lrw_filter100
#----------------------------------------------##

#----------------------------------------------##
python test_psnr_ssim_lrw.py --save_name ${name}_lrw_filter100 --bool_crop_and_align False --start_frame_id ${start_frame} --fake_pth "../talking_head_testing/25fps_video/align_crop/${name}_lrw_filter100/*.mp4" &
#----------------------------------------------##

#----------------------------------------------##
CUDA_VISIBLE_DEVICES=${device} python test_fid_lrw.py --save_name ${name}_lrw_filter100 --fake_pth "../talking_head_testing/25fps_video/align_crop/${name}_lrw_filter100/*.mp4" &
#----------------------------------------------##

#----------------------------------------------##
python test_lmd_lrw.py --save_name ${name}_lrw_filter100 --fake_pth "../talking_head_testing/25fps_video/align_crop/${name}_lrw_filter100/*.mp4" &
#----------------------------------------------##

wait

#----------------------------------------------##
CUDA_VISIBLE_DEVICES=${device} python test_sync_conf.py --save_name ${name}_lrw_filter100 --fake_pth "../talking_head_testing/25fps_video/pcavs_crop/${name}_lrw_filter100/*.mp4" --tmp_dir temps/lastversion_lrw/${name}_lrw_filter100 --log_rt results_lastversion_lrw
#----------------------------------------------##

wait

cd "../"
echo " ----------------------------------------------------- " >> lrw_res
echo "${name} psnr start from ${start_frame} frame" >> lrw_res
tail -n 1 "./code/results_lastversion_lrw/${name}_lrw_filter100.txt" >> lrw_res
tail -n 5 "./code/result_psnr_lrw/${name}_lrw_filter100.txt" >> lrw_res
tail -n 5 "./code/results_lrw/${name}_lrw_filter100.txt" >> lrw_res
tail -n 8 "./code/result_lrw/${name}_lrw_filter100.txt" >> lrw_res
echo "${name}" >> lrw_res
echo " ----------------------------------------------------- " >> lrw_res
