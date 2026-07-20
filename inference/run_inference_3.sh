python inference_mmseg_shapefile.py \
  --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/segformer_mit-b5_8x1_256x256_40k_queimadas_new.py" \
  --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/best_mIoU_iter_12000.pth" \
  --shp "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/Area_6_tif_inteiro/shp/2021_09_07_Planet.shp" \
  --tif "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/Area_6_tif_inteiro/tifs/2021_09_07_Planet.tif" \
  --out-dir "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/predictions/teste_recam" \  
  --device "cuda:1"
#
python inference_mmseg_shapefile.py \
  --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/segformer_mit-b5_8x1_256x256_40k_queimadas_new.py" \
  --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/best_mIoU_iter_12000.pth" \
  --shp "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/area1/shp/20211029_111837_0f2b_3B_AnalyticMS_SR.shp" \
  --tif "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/area1/tifs/area_1.tif" \
  --out-dir "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/predictions/teste_recam" \
  --device "cuda:1"
#
python inference_mmseg_shapefile.py \
  --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/segformer_mit-b5_8x1_256x256_40k_queimadas_new.py" \
  --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/best_mIoU_iter_12000.pth" \
  --shp "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/area3/shp/20210928_131659_63_242d_3B_AnalyticMS_SR.shp" \
  --tif "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/area3/tifs/area_3.tif" \
  --out-dir "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/predictions/teste_recam" \
  --device "cuda:1"
#
python inference_mmseg_shapefile.py \
  --config "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/segformer_mit-b5_8x1_256x256_40k_queimadas_new.py" \
  --checkpoint "/mnt/58B0FAA2B0FA85B2/Biomas/mmsegmentation/work_dirs/queimadas/wsss_segformer/recam/best_mIoU_iter_12000.pth" \
  --shp "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/area7_filtrado/shp/20211005_130333_30_241b_3B_AnalyticMS_SR_harmonized_clipSHP.shp" \
  --tif "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/nova_versao/area7_filtrado/tifs/area_7.tif" \
  --out-dir "/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/predictions/teste_recam" \
  --device "cuda:1"

