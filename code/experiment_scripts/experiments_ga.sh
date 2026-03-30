models=( 0 1 2 3 4 5 6 7 8 9 )

datasets=(
    ./data/detests-dis/
    ./data/exist/
    ./data/hate_speech/
    ./data/homo-mex-2023/
)

for model in "${models[@]}"
do
    for data_path in "${datasets[@]}"
    do
        python ga_predict.py -m $model -t 0 -d $data_path
    done
done