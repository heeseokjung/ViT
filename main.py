import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchsummary import summary
from model.transformer import VisionTransformer
from training_wrapper import TrainingWrapper

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pl.seed_everything(42)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    '''
    dataiter = iter(trainloader)
    imgs, labels = dataiter.next()
    print(f'img shape: {imgs.shape}')
    print(f'lables shape: {labels.shape}')
    print(labels)
    '''

    # 모델 초기화
    model = VisionTransformer()
    model = model.to(device)
    model = TrainingWrapper(model, lr=0.05)

    # trainer 초기화
    from pytorch_lightning.profiler import SimpleProfiler
    profiler = SimpleProfiler()
    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1,
        profiler=profiler
    )
    
    # trainer.fit
    trainer.fit(model, trainloader, valloader)

if __name__ == "__main__":
    main()