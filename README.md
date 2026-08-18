This is a project to do next-token prediction with RL.
The idea is to let the model 'think' before it predicts sequences like it does in pretraining.
The general gist will be to train a base model to learn to use think XML tags to do its internal reasoning before predicting a continuation.
The pipeline will look something like
1) teacher explanations from gemma4-31b to show the small model what thinking tags look like and how to format it. use vllm to get the teacher predicted continuations.
2) train a lora on a smaller model by doing sft on the teacher predictions (it should hopefully learn to use thinking tags and do step-by-step logical reasoning, not sure how much the specific details of how the teacher summaries look matters)
3) train it on its own on-policy reasoning, using the GRPO machinery
hopefully this can improve reasoning in the sort of mid-training stage? either way, pretty fun idea!