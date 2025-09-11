import torch.nn as nn
import torchvision.models as models

def get_model(model_name='resnet18', pretrained=True):
    """
    一个灵活的函数，可以根据名字获取不同的预训练模型。
    """
    if model_name == 'resnet18':
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet18(weights=weights)
    # 以后我们想试ResNet34，只需要在这里加几行代码
    # elif model_name == 'resnet34':
    #     model = models.resnet34(...)
    
    # 替换最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
        
    print(f"✅ {model_name} model is ready for fine-tuning.")
    return model