sizes=(8 16 32)
runs=(0 1 2 3 4 5 6 7 8 9)
for SIZE in ${sizes[@]}; do
    for RUN in ${runs[@]}; do
        python -u multitask_train.py --n_epochs=200 --max_len=50 --max_sent=50 --train_size_support=$SIZE --train_size_query=$SIZE --shots=$SIZE --lr 1e-3 > logs-max/out.$SIZE.$RUN.multitask 2>&1
    done;
done;
