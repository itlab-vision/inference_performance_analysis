@th_a = (1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 36, 48, 96);
@bs_a = (1, 48, 96, 192, 384);
$ni = 1152

$path_to_set = "../datasets/pascal_voc/";
$path_to_models = "../models";
$name_model = "ssd300";
$path_result = "./result_sync_detection";
$lib_extension = "~/panfsLink/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so";

foreach $bs (@bs_a)
{
 system"(python3 openvino_benchmark_sync.py -t detection -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -of $path_result/ -r $path_result/result_sync_deteAVX512.csv -s 1.0 -w 300 -he 300 -b $bs -e $lib_extension)";
 foreach $th (@th_a)
 {
  print "$bs $th n";
  system"(python3 openvino_benchmark_sync.py -t detection -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -of $path_result/ -r $path_result/result_sync_deteAVX512.csv -s 1.0 -w 300 -he 300 -b $bs -tn $th -e $lib_extension)";
 }
}