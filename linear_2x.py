import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from thop import profile, clever_format
import utils
from model import Model
class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # Create a model to get the architecture
        model=Model()
        self.net = model.net
        self.fc1 = nn.Linear(4096, num_class, bias=True)

        # Load pretrained weights
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')['resnet']
        model_dict = self.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k[len("net."):]: v for k, v in pretrained_dict.items() if k.startswith("net.")}

        # Update only matching layers
        for key in model_dict.keys():
            if key in pretrained_dict and model_dict[key].shape == pretrained_dict[key].shape:
                model_dict[key] = pretrained_dict[key]

        # Load the modified state dict
        self.load_state_dict(model_dict, strict=False)

    def forward(self, x):
        x = self.net(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc1(feature)
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    
    train_data = CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    
    for param in model.net.parameters():
        param.requires_grad = False

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    
    optimizer = torch.optim.SGD(model.fc1.parameters(), lr=0.01, momentum=0.9, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_criterion = nn.CrossEntropyLoss()
    
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        # Step the learning rate scheduler
        scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/linear_statistics.csv', index_label='epoch')
        
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/linear_model.pth')
