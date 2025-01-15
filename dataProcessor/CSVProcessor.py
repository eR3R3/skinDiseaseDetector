import os
import pandas as pd

train_folder = "/root/autodl-tmp/dataset/skinDisease_split/train"
test_folder = "/root/autodl-tmp/dataset/skinDisease_split/test"
val_folder = "/root/autodl-tmp/dataset/skinDisease_split/val"
csv_file = "/root/code_file/MedMamba/metadata.csv"
data = pd.read_csv(csv_file)

train_data = pd.DataFrame(columns=data.columns)
test_data = pd.DataFrame(columns=data.columns)
val_data = pd.DataFrame(columns=data.columns)
classes={"ACK","BCC","MEL","NEV","SCC","SEK"}
for _, row in data.iterrows():
    file_name = row["img_id"]
    for i in classes:
        if os.path.exists(train_folder+"/" + i+"/"+file_name):
            train_data = pd.concat([train_data, pd.DataFrame([row])], ignore_index=True)
        elif os.path.exists(test_folder+"/" + i+"/"+file_name):
            test_data = pd.concat([test_data, pd.DataFrame([row])], ignore_index=True)
        elif os.path.exists(val_folder+"/" + i+"/"+file_name):
            val_data = pd.concat([val_data, pd.DataFrame([row])], ignore_index=True)


train_data.to_csv("/root/autodl-tmp/dataset/skinDisease_split/train.csv", index=False)
test_data.to_csv("/root/autodl-tmp/dataset/skinDisease_split/test.csv", index=False)
val_data.to_csv("/root/autodl-tmp/dataset/skinDisease_split/val.csv", index=False)


