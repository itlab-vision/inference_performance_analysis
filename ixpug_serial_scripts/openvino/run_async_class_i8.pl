@th_a = (12, 24, 36, 48, 96);
@bs_a = (1, 2, 8, 12, 24);
@rn_a = (2, 4, 8, 24, 48);
$ni = 1152;

$path_to_set = "../datasets/imagenet/";
$path_to_models = "../models";
$name_model = "resnet-50_i8";
$path_result = "./result_async_classification_i8";

foreach $bs (@bs_a)
{
 foreach $rn (@rn_a)
 {
  system"(python3 openvino_benchmark_async.py -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -o False -of $path_result/ -r $path_result/result_async.csv -s 1.0 -w 224 -he 224 -b $bs -rn $rn)";
  foreach $sn (@rn_a)
  {
   system"(python3 openvino_benchmark_async.py -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -o False -of $path_result/ -r $path_result/result_async.csv -s 1.0 -w 224 -he 224 -b $bs -rn $rn -sn $sn)";
  }
  foreach $th (@th_a)
  {
   foreach $sn (@rn_a)
   {
    print "$bs $rn $th $sn \n";
    system"(python3 openvino_benchmark_async.py -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -o False -of $path_result/ -r $path_result/result_async.csv -s 1.0 -w 224 -he 224 -b $bs -sn $sn -rn $rn -tn $th)";
   }
  }
 }
}
