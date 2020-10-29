@th_a = (1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 36, 48, 96);
%aff_a = (
"scatter"=>"granularity=fine,scatter"
,"compact_1_0"=>"granularity=fine,compact,1,0");

@bs_a = (1, 48, 96, 192, 384);
$ni = 1152;

$path_to_set = "../datasets/imagenet/";
$path_to_models = "../models";
$name_model = "resnet-50";
$path_result = "./result_classification";

foreach $bs (@bs_a)
{
 foreach $th (@th_a)
 {
  system"(export OMP_NUM_THREADS=$th; python3 caffe_benchmark.py -t classification -i $path_to_set -p $path_to_models/$name_model.prototxt -m $path_to_models/$name_model.caffemodel -ni $ni -o False -b $bs -of $path_result/ -r $path_result/result.csv)";
 }
 foreach $key (keys %aff_a)
 {
  $aff = $aff_a{$key};
  foreach $th (@th_a)
  {
   system"(export KMP_AFFINITY=$aff ;export OMP_NUM_THREADS=$th; python3 caffe_benchmark.py -t classification -i $path_to_set -p $path_to_models/$name_model.prototxt -m $path_to_models/$name_model.caffemodel -ni $ni -o False -b $bs -of $path_result/ -r $path_result/result_aff_${key}.csv)";
  }
 }
}