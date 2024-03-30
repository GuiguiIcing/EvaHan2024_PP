data="training_data.txt"
pred_data_file="EvaHan2024_testset.txt"
pred_data_file_zz="EvaHan2024_testsetzz.txt"
pred_path="result.a.txt"
pred_path_zz="result.zz.a.txt"

feat="SIKU-BERT"
method="blstm.crf"


batchsize=512
# 2024.3.2
# Two crfs
# predict on Test A set
CUDA_VISIBLE_DEVICES=4 nohup python -u run.py predict \
    -p \
    --feat=$feat \
    --data=dataset/$data \
    --pred_data=dataset/$pred_data_file\
    --pred_path=TestPredict/$pred_path \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.gram.nopunc.clr.llm \
    > log/pred/$feat.$method.gram.llm.nopunc.pred.clr.a.log 2>&1 &

# predict on zz test set (Test B)
CUDA_VISIBLE_DEVICES=5 nohup python -u run.py predict \
    -p \
    --feat=$feat \
    --data=dataset/$data \
    --pred_data=dataset/$pred_data_file_zz\
    --pred_path=TestPredict/$pred_path_zz \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.gram.nopunc.clr.llm \
    > log/pred/$feat.$method.gram.llm.nopunc.pred.clr.zz.a.log 2>&1 &
