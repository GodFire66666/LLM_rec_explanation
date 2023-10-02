from recbole.quick_start import load_data_and_model
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='experiments/ml-100k/BPR/BPR-Aug-22-2023_18-21-27.pth',
)
print(model)