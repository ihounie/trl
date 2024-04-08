for resilient_alpha in 1.0 100.0 0.01
do
    for dual_lr in 1.0 10.0 0.1
    do
        for tolerance in 0.8 0.5 0.0
        do
            python dpof.py --lr_scheduler_type "cosine" --beta 0.1 --max_prompt_length 1024 --max_length 1536  --learning_rate 5e-5 --algorithm feasible --optim paged_adamw_32bit --dataset orca --train_epochs 10 --model_name_or_path=teknium/OpenHermes-2.5-Mistral-7B --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --logging_steps 100 --eval_steps 500 --output_dir=dpo_intel --warmup_steps 200 --report_to wandb --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=16 --lora_alpha=16 --loss_tolerance $tolerance --dual_lr $dual_lr --resilient_alpha $resilient_alpha
        done
    done
done