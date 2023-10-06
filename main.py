from sklearn.model_selection import train_test_split

from framework import CHARLEE
from utils import ScaleData

from torch.utils.data.sampler import SubsetRandomSampler
from channel_priority.kmeans import kmeans
# from channel_priority.classelbow import ElbowPair
# from channel_priority.elbow import elbow
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime
import sys

import wandb

# Initialize configs

model_config = {
    "batch_size": 32,  # Small batch size because checkpoint rewards collect a lot of actions
    "n_epochs": 250,  # Number of training epochs
    'val_perc': 0.2,  # Validation percentage of data
    "learning_rate": 0.001,  # Learning rate for optimizer
    "earliness_weight_factor": 0.9,  # 0 to 1, float,with 1 giving more weight to earliness reward and 0 to accuracy
    'inception_depth': 2,
    "num_filters_per_channel": 5,  # Option for the hidden state convolutional network
    "kernel_size": 9,  # Option for the hidden state convolutional network
    "num_feats_per_filter": 6,
    # Max, min, mean, percentage of positive values, mean of positive values and mean of indexes of positive values
    "num_channel_slices": 10,  # Split the input channels in groups
    "num_checkpoints": 3,  # N checkpoints split the time series in N+1 parts

    "n_hidden_layers": 4,  # Hidden layers of the policy filter network
    "n_hidden_layer_units": 30,  # Hidden layer units of the policy filter network
    "policy_nonlinear": torch.nn.Tanh,  # Activation function of policy filter network
    'policy_use_dropout': False,  # Boolean choice of dropout usage in policy filter network
    'policy_dropout_perc': 0.3,  # Percentage of dropout (if used) in policy filter network

    "baseline_n_hidden_layers": 4,  # Hidden layers of the baseline network
    "baseline_n_hidden_layer_units": 30,  # Hidden layer units of the baseline network
    "baseline_nonlinear": torch.nn.Tanh,  # Activation function of the baseline network
    'baseline_use_dropout': False,  # Boolean choice of dropout usage in the baseline network
    'baseline_dropout_perc': 0.3,  # Percentage of dropout (if used) in the baseline network

    ## The 2 options below have not been utilized in the paper experiments
    "classifier_use_input_dropout": False,  ## Boolean choice of dropout usage in input of final classifier
    'classifier_input_dropout_perc': 0.3,  # Percentage of dropout (if used) in input of final classifier

    'datetime': datetime.now().strftime("%Y%m%d-%H%M"),

    ## The choices below have not been utilized in the paper experiments
    "lacc_mult": 1,  # Multipliers for losses

    'use_dl_model': 'resnet',
    ## if this is set to 'inceptiontime' or 'resnet', the respective model will be used

    'pretrained': False,  ##This refers to the whole framework, to avoid training any part of it
    'pretrained_framework_path': '',

    "random_seed": 3,
    "num_random_tests": 25,  ##Since agent is probabilistic, we run multiple tests
    "channel_priority_method": kmeans(),
    'use_priorities': True,
    ## Apply the priorities that are calculated by the above method by changing the order of channels
    'reverse_channel_priorities': True,
    # This is just an implementation choice for simplicity of implementation, the channels are abandoned in reverse order

    ## The filtering policy can also be implemented with a normal distribution, but this choice has not been used in the experiments
    'filter_policy_std': 0.1,  ## For the normal filter policy, we need a standard deviation (the mu is predicted)

    ## Preprocessing method and dimension of preprocessing of multivariate data, see https://github.com/lpphd/mtsscaling
    'data_scaling_method': 'standard',
    'data_scaling_dim': 'channels',

    ##These options have not been used in the paper experiments, but they can be used to freeze training of some parts of the framework
    ## during either the beginning or final epochs
    'warmup_epochs': 0,  ## Only train classifiers for these epochs at beginning of training
    'cooldown_epochs': 0,  ## Only train classifiers for these epochs at end of training

    'discount_rewards': True,  ## Discount the future rewards
    ## The filtering policy can also be implemented with a normal or discrete distribution, but this choice has not been used in the experiments
    'filter_distribution': 'beta',  # beta or normal
    "notes": ""
}


def calculate_channel_priority(train_x, train_y, num_checkpoints, method):
    """
    Calculates the channel priority as described in the paper and returns the appropriate order, from most to least
    useful channel
    """
    num_channels = train_x.shape[1]
    timesteps = train_x.shape[-1]
    orderings = np.zeros((num_checkpoints + 1, num_channels))
    slice_length = timesteps // (num_checkpoints + 1)
    mask_value = 0
    for i in range(num_checkpoints + 1):
        endpoint = (i + 1) * slice_length
        if i == num_checkpoints:
            endpoint *= 2
        mask = np.zeros_like(train_x).astype('bool')
        mask[:, :, :endpoint] = True
        masked_train_x = train_x * mask + mask_value * np.ones_like(train_x) * (~mask)
        train_df = from_3d_numpy_to_nested(masked_train_x)
        method.fit(train_df, train_y)
        orderings[i] = method.ranked_dims

    scores = (np.argsort(orderings[:, ::-1], axis=-1) + 1) * np.linspace(1, 0.1, num_checkpoints + 1).reshape((-1, 1))
    scores = scores.sum(axis=0)
    final_priority = np.argsort(scores)[::-1]
    return final_priority


def calculate_earliness_metrics(test_filter_histories, timesteps, channels):
    """
            Given the history of the framework actions and information about the dataset format, calculate
            earliness metrics across all test samples, such as average percentage of input saved, median, minimum, etc.
    """
    fh = torch.cat(test_filter_histories)
    unique_configurations = fh.unique(dim=0)
    diffs = fh[:, 1:] - fh[:, :-1]
    policy_violating_samples_idx = (diffs > 0).sum(-1) > 0
    n_policy_violating_samples = ((diffs > 0).sum(-1) > 0).sum()
    timesteps_per_sample = channels * timesteps
    slice_length = timesteps // (fh.shape[-1])

    fh[:, :-1] = torch.ceil(fh[:, :-1] * channels).int() * slice_length
    fh[:, -1] = torch.ceil(fh[:, -1] * channels).int() * (timesteps - slice_length * (fh.shape[-1] - 1))
    fh = fh.sum(-1)
    fh = (timesteps_per_sample - fh) / timesteps_per_sample

    average_saved_timesteps_perc = fh.mean().item()
    std_saved_timesteps_perc = fh.std().item()
    min_saved_timesteps_perc = fh.min().item()
    max_saved_timesteps_perc = fh.max().item()
    median_saved_timesteps_perc = fh.median().item()

    return average_saved_timesteps_perc, std_saved_timesteps_perc, median_saved_timesteps_perc, min_saved_timesteps_perc, max_saved_timesteps_perc, n_policy_violating_samples, torch.cat(
        test_filter_histories)[policy_violating_samples_idx, :], unique_configurations.shape[0]


def prepare_data_loaders(filename, model_config):
    """
    Prepare dataset by loading iit from file, splitting into train, validation and test, scaling appropriately according to method and dimension,
    and applying the channel priority reordering, as described in the paper.
    """
    data = np.load(filename)
    dev = model_config['device']
    train_x, test_x = data['train_x'].astype(np.float32), data['test_x'].astype(np.float32)
    train_y, test_y = data['train_y'].astype(np.int64), data['test_y'].astype(np.int64)
    if str(model_config['val_perc']) == 'test_size':
        val_perc = test_x.shape[0] / train_x.shape[0]
    else:
        val_perc = model_config['val_perc']
    if val_perc > 0:
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_perc,
                                                          random_state=model_config['random_seed'],
                                                          stratify=train_y)
        _, val_x = ScaleData(train_x, val_x, model_config['data_scaling_method'], model_config['data_scaling_dim'], 0)
    train_x, test_x = ScaleData(train_x, test_x, model_config['data_scaling_method'], model_config['data_scaling_dim'],
                                0)
    priorities = calculate_channel_priority(train_x, train_y, model_config['num_checkpoints'],
                                            model_config['channel_priority_method'])
    if model_config['reverse_channel_priorities']:
        priorities = priorities[::-1]
    if model_config['use_priorities']:
        train_x, test_x = train_x[:, priorities, :], test_x[:, priorities, :]
        if val_perc > 0:
            val_x = val_x[:, priorities, :]

    mask_value = 0

    train_x, test_x = torch.from_numpy(train_x).to(dev), torch.from_numpy(test_x).to(dev)
    train_y, test_y = torch.from_numpy(train_y).to(dev), torch.from_numpy(test_y).to(dev)

    # --- get data loaders ---
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=model_config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y),
                                              batch_size=model_config["batch_size"])

    if val_perc > 0:
        val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_x, val_y),
                                                 batch_size=model_config["batch_size"])
    else:
        val_loader = None

    data_config = {
        "timesteps": train_x.shape[2],  #
        "channels": train_x.shape[1],
        "n_classes": np.unique(train_y.cpu()).size
    }

    data_config['mask_value'] = mask_value
    return train_loader, val_loader, test_loader, data_config


if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_config['random_seed'] = int(sys.argv[1])

    if model_config['pretrained']:
        model_config['random_seed'] = int(model_config['pretrained_framework_path'].split(".")[0].split("_")[-1])

    torch.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    torch.autograd.set_detect_anomaly(True)

    ## Set dataset directory and name here
    data_dir = "Datasets/"

    model_config['dataset'] = 'SyntheticDatasetEE.npz'

    filename = data_dir + model_config['dataset']

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        dev = torch.device("cpu")

    print(dev, flush=True)
    print("=" * 20, flush=True)

    model_config['device'] = dev

    train_loader, val_loader, test_loader, data_config = prepare_data_loaders(filename, model_config)

    ## Number of channel groups is the minimum number between the ones seleced in config and the number of dataset channels
    model_config['num_channel_slices'] = min(model_config['num_channel_slices'], data_config['channels'])
    ## Set up slice length dependent on dataset timesteps, to acommodate various dataset sizes
    slice_length = data_config['timesteps'] // (model_config['num_checkpoints'] + 1)
    model_config['kernel_size'] = max(3, slice_length // 3)

    ## Log experiments using Wandb
    wandb.init(project="project_name",
               name=f"experiment_name",
               entity="entity", group="group_name", config=model_config)
    wandb.run.log_code(".")

    if not model_config['pretrained']:
        model_config['pretrained_framework_path'] = f"policy_training_{wandb.run.id}_{model_config['random_seed']}.pth"

    print(model_config, flush=True)

    model = CHARLEE(model_config, data_config).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])

    wandb.watch(model, log='all', log_freq=model_config['batch_size'])
    if model_config['pretrained']:
        model.load_state_dict(torch.load(model_config['pretrained_framework_path']))

    else:
        ## Typical training loop
        training_loss = []
        training_acc_loss = []
        training_pol_loss = []
        training_rewards = []
        min_train_loss = np.inf
        min_val_score = -np.inf

        for epoch in range(model_config["n_epochs"]):
            train_loss = 0
            train_rewards = 0
            train_acc_loss = 0
            train_pol_stop_loss = 0
            train_pol_filter_loss = 0
            train_value_baseline_loss = 0
            model.train()
            train_filter_histories = []
            predictions = []
            labels = []
            for i, (X, y) in enumerate(train_loader):
                logits, filter_history = model(X, epoch=epoch)
                train_filter_histories.append(filter_history)
                Lacc, Lstop, Lfilter, Lbaseline, Reward = model.computeLoss(logits, y)
                loss = Lacc + Lstop + Lfilter + Lbaseline
                y_hat = torch.softmax(logits, 1)
                predictions.extend(y_hat.detach().tolist())
                labels.extend(y.tolist())
                train_loss += loss.item()
                train_rewards += Reward.item()
                train_acc_loss += Lacc.item()
                train_pol_stop_loss += Lstop.item()
                train_pol_filter_loss += Lfilter.item()
                train_value_baseline_loss += Lbaseline.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            earliness_metrics = calculate_earliness_metrics(train_filter_histories, data_config['timesteps'],
                                                            data_config['channels'])

            predicted_classes = np.array(predictions).argmax(-1)
            acc = accuracy_score(np.array(labels), predicted_classes)
            f1 = f1_score(np.array(labels), np.array(predicted_classes), average=
            'weighted')

            print(f'Epoch [{epoch + 1}/{model_config["n_epochs"]}] - Training Loss {train_loss}', flush=True)

            wandb.log({"train_loss": train_loss}, step=epoch)
            wandb.log({"train_acc_loss": train_acc_loss}, step=epoch)
            wandb.log({"train_acc": acc}, step=epoch)
            wandb.log({"train_f1": f1}, step=epoch)
            wandb.log({"train_pol_stop_loss": train_pol_stop_loss}, step=epoch)
            wandb.log({"train_pol_filter_loss": train_pol_filter_loss}, step=epoch)
            wandb.log({"train_baseline_loss": train_value_baseline_loss}, step=epoch)
            wandb.log({"train_rewards": train_rewards}, step=epoch)

            wandb.log({"train_av_perc_saved": earliness_metrics[0]}, step=epoch)
            wandb.log({"train_avstd_perc": earliness_metrics[1]}, step=epoch)
            wandb.log({"train_median_perc": earliness_metrics[2]}, step=epoch)
            wandb.log({"train_min_perc": earliness_metrics[3]}, step=epoch)
            wandb.log({"train_max_perc": earliness_metrics[4]}, step=epoch)
            wandb.log({"unique_configurations": earliness_metrics[7]}, step=epoch)

            val_filter_histories = []
            val_labels = []
            val_predictions = []
            val_loss = 0
            val_rewards = 0
            val_acc_loss = 0
            val_pol_stop_loss = 0
            val_pol_filter_loss = 0
            val_value_baseline_loss = 0
            model.eval()
            for i, (X_val, y_val) in enumerate(val_loader):
                logits, filter_history = model(X_val, epoch=epoch, test=True)
                val_filter_histories.append(filter_history)
                Lacc, Lstop, Lfilter, Lbaseline, Reward = model.computeLoss(logits, y_val)
                loss = Lacc + Lstop + Lfilter + Lbaseline
                val_y_hat = torch.softmax(logits, 1)
                val_predictions.extend(val_y_hat.tolist())
                val_labels.extend(y_val.tolist())
                val_loss += loss.item()
                val_rewards += Reward.item()
                val_acc_loss += Lacc.item()
                val_pol_stop_loss += Lstop.item()
                val_pol_filter_loss += Lfilter.item()
                val_value_baseline_loss += Lbaseline.item()

            earliness_metrics = calculate_earliness_metrics(val_filter_histories, data_config['timesteps'],
                                                            data_config['channels'])

            val_y_pred = np.array(val_predictions).argmax(-1)
            val_y_true = np.array(val_labels)
            acc = accuracy_score(val_y_true, val_y_pred)
            f1 = f1_score(val_y_true, val_y_pred, average=
            'weighted')

            wandb.log({"val_loss": val_loss}, step=epoch)
            wandb.log({"val_acc_loss": val_acc_loss}, step=epoch)
            wandb.log({"val_acc": acc}, step=epoch)
            wandb.log({"val_f1": f1}, step=epoch)
            wandb.log({"val_pol_stop_loss": val_pol_stop_loss}, step=epoch)
            wandb.log({"val_pol_filter_loss": val_pol_filter_loss}, step=epoch)
            wandb.log({"val_rewards": val_rewards}, step=epoch)
            wandb.log({"val_av_perc_saved": earliness_metrics[0]}, step=epoch)
            wandb.log({"val_avstd_perc": earliness_metrics[1]}, step=epoch)
            wandb.log({"val_median_perc": earliness_metrics[2]}, step=epoch)
            wandb.log({"val_min_perc": earliness_metrics[3]}, step=epoch)
            wandb.log({"val_max_perc": earliness_metrics[4]}, step=epoch)
            wandb.log({"val_unique_configurations": earliness_metrics[7]}, step=epoch)
            wandb.log({"val_baseline_loss": val_value_baseline_loss}, step=epoch)

            val_score = f1 * (1 - model_config['earliness_weight_factor']) + model_config[
                'earliness_weight_factor'] * earliness_metrics[0]

            ## Save weight with best validatioon performance according to earliness factor
            if val_score > min_val_score:
                torch.save(model.state_dict(), model_config['pretrained_framework_path'])
                min_val_score = val_score

    model.load_state_dict(torch.load(model_config['pretrained_framework_path']))
    model.eval()
    test_accs = []
    test_f1s = []
    test_av_percs = []
    test_avp_stds = []
    test_median_percs = []
    test_min_percs = []
    test_max_percs = []
    test_unique_confs = []
    test_policy_violating_samples = []

    ## Run multiple random tests to get better performance estimation, due to stochastic nature of framework
    for j in range(model_config['num_random_tests']):
        torch.manual_seed(j)
        np.random.seed(j)
        test_filter_histories = []
        predictions = []
        labels = []
        with torch.no_grad():
            for i, (X_test, y_test) in enumerate(test_loader):
                logits, filter_history = model(X_test, test=True)
                model.computeLoss(logits, y_test)
                test_filter_histories.append(filter_history)
                y_hat_test = torch.softmax(logits, 1)
                predictions.extend(y_hat_test.tolist())
                labels.extend(y_test.tolist())

        y_pred = np.array(predictions).argmax(-1)
        y_true = np.array(labels)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=
        'weighted')
        earliness_metrics = calculate_earliness_metrics(test_filter_histories, data_config['timesteps'],
                                                        data_config['channels'])

        test_accs.append(acc)
        test_f1s.append(f1)
        test_av_percs.append(earliness_metrics[0])
        test_avp_stds.append(earliness_metrics[1])
        test_median_percs.append(earliness_metrics[2])
        test_min_percs.append(earliness_metrics[3])
        test_max_percs.append(earliness_metrics[4])
        test_unique_confs.append(earliness_metrics[7])
        test_policy_violating_samples.append(earliness_metrics[5].item())

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_accs)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_accs": wandb.plot.line(table, "x", "y",
                                            title="Test accuracies")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_f1s)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_f1s": wandb.plot.line(table, "x", "y",
                                           title="Test F1s")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_av_percs)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_av_percs": wandb.plot.line(table, "x", "y",
                                                title="Test average percentages")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_avp_stds)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_avp_stds": wandb.plot.line(table, "x", "y",
                                                title="Test average percentage stds")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_median_percs)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_median_percs": wandb.plot.line(table, "x", "y",
                                                    title="Test median percentages")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_min_percs)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_min_percs": wandb.plot.line(table, "x", "y",
                                                 title="Test min percentages")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_max_percs)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_max_percs": wandb.plot.line(table, "x", "y",
                                                 title="Test max percentages")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_unique_confs)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_unique_confs": wandb.plot.line(table, "x", "y",
                                                    title="Test unique confs")})

    data = [[x, y] for (x, y) in zip(np.arange(model_config['num_random_tests']), test_policy_violating_samples)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"policy_violating_samples": wandb.plot.line(table, "x", "y",
                                                           title="Policy violating samples")})
