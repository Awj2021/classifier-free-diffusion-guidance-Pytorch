from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
from cifar import CIFAR10
from torch.utils.data.distributed import DistributedSampler

def load_data(batchsize:int, numworkers:int, noisy_type:str, noise_path:str, is_human:bool) -> tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root = './',
                        train = True,
                        download = True,
                        noise_type = noisy_type,
                        noise_path = noise_path,
                        is_human = is_human,
                        transform = trans
                    )
    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        sampler = sampler,
                        drop_last = True
                    )
    return data_train, trainloader, sampler

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
