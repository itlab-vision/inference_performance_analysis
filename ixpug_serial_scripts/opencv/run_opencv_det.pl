@th_a = (1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 36, 48, 96);
@bs_a = (1, 48, 96, 192, 384);
$ni = 1152;

$path_to_set = "../datasets/pascal_voc/";
$path_to_models = "../models";
$name_model = "ssd300";
$path_result = "./results_detection";

foreach $bs (@bs_a)
{
 foreach $th (@th_a)
 {
  system"(export TBB_NUM_THREADS=$th; export OMP_NUM_THREADS=$th; python3 opencv_benchmark.py -t detection -i $path_to_set -p $name_model/$name_model.prototxt -m $name_model/$name_model.caffemodel -ni $ni -o False -b $bs -of $path_result/ -r $path_result/result.csv -w 300 -he 300 -s 1.0)";
 }
}