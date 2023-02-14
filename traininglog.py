import pandas as pd
import numpy as np
import os

def log_training(losses):
    epochs = []
    for i in range(len(losses)):
      epochs.append(i+1)
    df = pd.DataFrame.from_dict({ 'epoch': epochs, 'loss': losses })

    dirs = os.listdir('./training_record')
    dir_name = f"./training_record/{len(dirs)}"
    os.mkdir(dir_name)

    df.to_csv(f"{dir_name}/training.csv")

def log_about(train_set, test_set, epochs, lr, loss, percentage, hidden_layer_nodes):
  path = "./training_record/record.csv"
  df = pd.read_csv(path)
  data = [[
    train_set,
    test_set,
    epochs,
    lr,
    loss,
    percentage,
    hidden_layer_nodes
  ]]
  # df = pd.DataFrame(np.insert(df.values, len(df.values), data, axis=0))
  new_df = pd.DataFrame(data, columns=[
    "train_set",
    "test_set",
    "epochs",
    "lr",
    "loss",
    "percentage",
    "hidden_layer_nodes"
  ])
  df = pd.concat([df, new_df], ignore_index=True)
  df.to_csv(path)