#python prediction_mmseg_iou.py \
#    --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/puzzle_r101/segformer_mit-b5_8x1_256x256_40k_queimadas_new_r101.py" \
#    --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/puzzle_r101/best_mIoU_iter_32000.pth" \
#    --split-list "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/dataset2_bkp/data/test_sem_interseccao.txt" \
#    --device "cuda:1"
#
#python prediction_mmseg_iou.py \
#    --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/puzzle_r101_sigma2/segformer_mit-b5_8x1_256x256_40k_queimadas_new_r01_sigma2.py" \
#    --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/puzzle_r101_sigma2/best_mIoU_iter_24000.pth" \
#    --split-list "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/dataset2_bkp/data/test_sem_interseccao.txt" \
#    --device "cuda:1"

python inference_mmseg_shapefile.py \
  --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/puzzle_r101_sigma2/segformer_mit-b5_8x1_256x256_40k_queimadas_new_r01_sigma2.py" \
  --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/puzzle_r101_sigma2/best_mIoU_iter_24000.pth" \
  --shp "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/Area_4/corrigida/Area_4_20211001_131224_70_2235_3B_AnalyticMS_SR.shp" \
  --tif "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/Area_4/files/Area_4_20211001_131224_70_2235_3B_AnalyticMS_SR.tif" \
  --out-dir "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/predictions/teste" \
  --device "cuda:1"