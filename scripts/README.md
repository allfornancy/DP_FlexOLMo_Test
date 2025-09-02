## Training scripts

| Script | Description |
| ------ | ----------- |
| [train_public_model.sh](train_public_model.sh) | Train a dense 7B model on the public mix and upcycle it to a 2x7B model |
| [train_expert_model.sh](train_expert_model.sh) | Train expert 2x7B model on domain data, using the public model as the seed | 
| [combine_experts.sh](combine_experts.sh) | Combine experts into a FlexOlmo model, and optionally perform router training on proxy data |