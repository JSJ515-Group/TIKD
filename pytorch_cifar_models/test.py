import pytorch_cifar_models.repvgg as repvgg
import torch

model_name = "repvgg_a2"
dataset = "cifar100"

#动态加载模型
model = getattr(repvgg, f"{dataset}_{model_name}")(pretrained=True)
state_dict = model.state_dict()
print(state_dict)