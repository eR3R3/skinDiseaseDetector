import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from mamba import VSSM as medmamba

from sklearn.preprocessing import OneHotEncoder, StandardScaler
class ImageFolderWithName(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        filename = os.path.basename(path)
        return sample, target, filename
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #photoDataProcessor
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = ImageFolderWithName(root="/root/autodl-tmp/dataset/skinDisease_split/train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = ImageFolderWithName(root="/root/autodl-tmp/dataset/skinDisease_split/val",
                                        transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    #textDataProcessor
    df_train = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/train.csv")
    df_val = pd.read_csv("/root/autodl-tmp/dataset/skinDisease_split/val.csv")

    # df_train.dropna(inplace=True)
    # df_val.dropna(inplace=True)

    df_train = df_train.iloc[:, 2:]
    df_val = df_val.iloc[:, 2:]

    X_train = df_train.drop(["diagnostic", "background_father", "background_mother", "elevation"], axis=1)

    X_val = df_val.drop(["diagnostic", "background_father", "background_mother", "elevation"], axis=1)

    numeric_features = ["age", "diameter_1", "diameter_2"]
    categorical_features = ["gender", "smoke", "drink", "region", "grew", "changed", "pesticide", "skin_cancer_history",
                            "cancer_history", "has_sewage_system", "fitspatrick", "itch", "hurt", "bleed", "biopsed",
                            "has_piped_water"]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    train_name_to_idx = {s: i for i, s in enumerate(X_train[:,-1])}
    X_train = X_train[:, :-1]
    val_name_to_idx = {s: i for i, s in enumerate(X_val[:,-1])}
    X_val = X_val[:, :-1]
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    #init
    input_dim = X_train.shape[1]
    hidden_dims = [128,64, 32]
    net = medmamba(MLP_input_dim=input_dim,MLP_hidden_dims=hidden_dims,num_classes=6)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format("MultiMedmambaLarge")
    train_steps = len(train_loader)

    #train
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels,filename = data
            text = np.full((images.shape[0], X_train.shape[1]), 0.0, dtype=np.float32)
            for i in range(images.shape[0]):
                j=filename[i]

                text[i]=X_train[train_name_to_idx[j]]

            text=torch.from_numpy(text).float()

            optimizer.zero_grad()
            outputs = net(images.to(device),text.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels,val_filename = val_data

                val_text = np.full((val_images.shape[0], X_val.shape[1]), 0.0, dtype=np.float32)
                for i in range(val_images.shape[0]):
                    j = val_filename[i]

                    val_text[i] = X_val[val_name_to_idx[j]]

                val_text = torch.from_numpy(val_text).float()

                outputs = net(val_images.to(device),val_text.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
