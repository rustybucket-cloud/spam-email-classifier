import pandas as pd
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

def log_about(train_set, test_set, epochs, lr, loss, percentage):
  path = "./training_record/record.csv"
  df = pd.read_csv(path)
  data = {
    "train_set": train_set,
    "test_set": test_set,
    "epochs": epochs,
    "lr": lr,
    "loss": loss,
    "percentage": percentage
  }
  df.append(data)
  df.to_csv(path)