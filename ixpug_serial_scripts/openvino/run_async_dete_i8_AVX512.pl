@th_a = (12, 24, 36, 48, 96);
@bs_a = (1, 2, 8, 12, 24);
@rn_a = (2, 4, 8, 24, 48);
$ni = 1152;

$path_to_set = "../datasets/pascal_voc/";
$path_to_models = "../models";
$name_model = "ssd300_i8";
$path_result = "./result_async_detection_i8";
$lib_extension = "~/panfsLink/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so";

foreach $bs (@bs_a)
{
 foreach $rn (@rn_a)
 {
  system"(python3 openvino_benchmark_async.py -t detection -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -of $path_result/ -r $path_result/result_asyncAVX512.csv -s 1.0 -w 300 -he 300 -b $bs -rn $rn -e $lib_extension)";
  foreach $sn (@rn_a)
  {
   system"(python3 openvino_benchmark_async.py -t detection -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -of $path_result/ -r $path_result/result_asyncAVX512.csv -s 1.0 -w 300 -he 300 -b $bs -rn $rn -sn $sn -e $lib_extension)";
  }
  foreach $th (@th_a)
  {
   foreach $sn (@rn_a)
   {
    print "$bs $rn $th $sn";
    system"(python3 openvino_benchmark_async.py -t detection -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -of $path_result/ -r $path_result/result_asyncAVX512.csv -s 1.0 -w 300 -he 300 -b $bs -sn $sn -rn $rn -tn $th -e $lib_extension)";
   }
  }
 }
}