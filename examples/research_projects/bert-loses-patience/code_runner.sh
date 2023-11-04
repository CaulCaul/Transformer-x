source ~/.venv/dgl/bin/activate

export GLUE_DIR=/data/glue_data_0
export TASK_NAME=MRPC

code=2

case $code in
  1)
    python ./run_glue_with_pabee.py \
      --model_type albert \
      --model_name_or_path albert-base-v2 \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir "$GLUE_DIR/$TASK_NAME" \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 1 \
      --per_gpu_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --save_steps 50 \
      --logging_steps 50 \
      --num_train_epochs 5 \
      --output_dir "~/code/save/bert-loses-patience/"$TASK_NAME \
      --overwrite_output_dir \
      --evaluate_during_training > ~/code/transformers/log_pabee_${TASK_NAME}_train.log
    ;;
  2)
    python ./run_glue_with_pabee.py \
      --model_type albert \
      --model_name_or_path albert-base-v2 \
      --task_name $TASK_NAME \
      --do_eval \
      --do_lower_case \
      --data_dir "$GLUE_DIR/$TASK_NAME" \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --logging_steps 50 \
      --num_train_epochs 15 \
      --output_dir "~/code/save/bert-loses-patience/"$TASK_NAME \
      --patience 3 > ~/code/transformers/log_pabee_${TASK_NAME}_eval_3.log
      # --eval_all_checkpoints \
      # --patience 3,4,5,6,7,8
    ;;
esac



