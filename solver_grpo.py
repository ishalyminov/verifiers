import torch
import verifiers as vf
from verifiers.utils.model_utils import get_model_and_tokenizer
from verifiers.trainers import grpo_defaults

vf_env = vf.load_environment(env_id="tau2_bench", domain="telecom", solo_mode=True)
model_name = f"Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = get_model_and_tokenizer(model_name, model_kwargs=dict(
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    use_cache=False,

))
model.gradient_checkpointing_enable()

training_args = grpo_defaults(run_name='solver-grpo')

training_args.per_device_train_batch_size = 1
training_args.num_generations = 2
training_args.gradient_accumulation_steps = 1
training_args.max_prompt_length = 14000
training_args.max_seq_len = 2048
training_args.eval_strategy = "steps"
training_args.eval_steps = 10
training_args.save_strategy = "steps"
training_args.save_steps = 100
training_args.max_steps = 200
training_args.eval_strategy = "steps"
training_args.eval_steps = 10
training_args.max_concurrent = 1
training_args.num_batches_ahead = 1

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()