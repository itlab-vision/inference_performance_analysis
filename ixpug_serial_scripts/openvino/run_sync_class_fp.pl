@th_a = (1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 36, 48, 96);
@bs_a = (1, 48, 96, 192, 384);
$ni = 1152;

$path_to_set = "../datasets/imagenet/";
$path_to_models = "../models";
$name_model = "resnet-50";
$path_result = "./result_sync_classification";

foreach $bs (@bs_a)
{
 system"(python3 openvino_benchmark_sync.py -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -o False -of $path_result/ -r $path_result/result_sync.csv -s 1.0 -w 224 -he 224 -b $bs)";
 foreach $th (@th_a)
 {
  print "$bs $th n";
  system"(python3 openvino_benchmark_sync.py -i $path_to_set -c $path_to_models/$name_model.xml -m $path_to_models/$name_model.bin -ni $ni -o False -of $path_result/ -r $path_result/result_sync.csv -s 1.0 -w 224 -he 224 -b $bs -tn $th)";
 }
}