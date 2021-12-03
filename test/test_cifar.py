"""
"""
import time
import oneflow as flow
import flowvision as vision
import oneflow.nn as nn
import oneflow.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(flow._C.relu(self.conv1(x)))
        x = self.pool(flow._C.relu(self.conv2(x)))
        x = flow.flatten(x, 1)  # flatten all dimensions except batch
        x = flow._C.relu(self.fc1(x))
        x = flow._C.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(epoch, num_workers, batch_size, print_interval):
    start = time.time()
    device = flow.device("cuda")
    net = Net()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    transform = vision.transforms.Compose(
        [
            vision.transforms.ToTensor(),
            vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_epoch = epoch
    data_dir = "./"

    trainset = vision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )
    trainloader = flow.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    final_loss = 0
    for epoch in range(1, train_epoch + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(dtype=flow.float32, device=device)
            labels = labels.to(dtype=flow.int64, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.cpu().detach().numpy()
            if i % print_interval == 0:
                final_loss = running_loss / print_interval
                print("epoch: %d  step: %5d  loss: %.3f " % (epoch, i, final_loss))
                running_loss = 0.0

    end = time.time()
    print(
        "epoch, num_worker, batch_size, print_interval >>>>>> ",
        epoch,
        num_workers,
        batch_size,
        print_interval,
        " final loss: ",
        final_loss,
        " cost: ",
        end - start,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", True)

    epoch = 1
    num_workers = 4
    print_interval = 200
    batch_size = 4
    test(epoch, num_workers, batch_size, print_interval)
