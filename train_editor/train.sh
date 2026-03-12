NUM=1
project_path="/data/li_yinghuang/EditCoT"
model_path="/data/li_yinghuang/EditCoT/models/Meta-Llama-3-8B-Instruct"
port_addr=11468


python $project_path/train_editor/train.py \
    --report_to "none" \
    --data_path $project_path/dataset_construct/output/llama3_dataset.json \
    --model_name_or_path $model_path \
    --output_dir $project_path/train_editor/output/output_llama_1gpu \
    --model_max_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy no \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --bf16 True