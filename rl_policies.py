from torch import nn
from utils import *
from torch.distributions import Beta, Normal
import torch
import torch.nn.functional as F


class RLPolicies(nn.Module):
    def __init__(self, model_config, data_config):
        super(RLPolicies, self).__init__()

        self.n_epochs = model_config["n_epochs"]
        self.timesteps = data_config["timesteps"]
        ## Dimensions of hidden state is determined by number of channels, filters per channel and features per filter
        ## In addition, as described in the paper, a float indicates the point in time of processing and the history of actions
        self.hidden_state_dim = 1 * (data_config['channels'] * model_config['num_filters_per_channel'] * model_config[
            'num_feats_per_filter']) + 1 + model_config['num_checkpoints'] + 1

        self.filter_floats = torch.linspace(0, 1, model_config['num_channel_slices'] + 1)

        self.n_channel_slices = model_config['num_channel_slices']
        self.earl_factor = model_config['earliness_weight_factor']
        self.num_checkpoints = model_config['num_checkpoints']

        self.filter_std = model_config['filter_policy_std']

        self.exploration_mode = model_config['exploration_mode']
        self.discount = model_config['discount_rewards']
        self.filter_dist = model_config['filter_distribution']

        ## Stop action threshold is calculated based on earliness factor
        self.stop_action_thresh = mapValue(0, 1, 1, 0.5, self.earl_factor)

        # --- Mappings ---
        if self.filter_dist == 'normal':
            filternetOut = 1
        elif self.filter_dist == 'beta':
            filternetOut = 2
        else:
            filternetOut = self.n_channel_slices + 1
        self.filterPolicyNet = createNet(self.hidden_state_dim, filternetOut,
                                         n_hidden_layers=model_config['n_hidden_layers'],
                                         n_hidden_layer_units=model_config['n_hidden_layer_units'],
                                         use_dropout=model_config['policy_use_dropout'],
                                         dropout_perc=model_config['policy_dropout_perc'],
                                         nonlinear=model_config['policy_nonlinear'])
        self.stopNet = createNet(self.hidden_state_dim + 1, 1,
                                 n_hidden_layers=model_config['n_hidden_layers'],
                                 n_hidden_layer_units=model_config['n_hidden_layer_units'],
                                 use_dropout=model_config['policy_use_dropout'],
                                 dropout_perc=model_config['policy_dropout_perc'],
                                 nonlinear=model_config['policy_nonlinear'])

        self.baselineNet = createNet(self.hidden_state_dim, 1,
                                     n_hidden_layers=model_config['baseline_n_hidden_layers'],
                                     n_hidden_layer_units=model_config['baseline_n_hidden_layer_units'],
                                     use_dropout=model_config['baseline_use_dropout'],
                                     dropout_perc=model_config['baseline_dropout_perc'],
                                     nonlinear=model_config['baseline_nonlinear']
                                     )

        ## Initialize network weights
        for l in self.filterPolicyNet:
            if l._get_name() == 'Linear':
                torch.nn.init.xavier_normal_(l.weight, gain=torch.nn.init.calculate_gain(
                    model_config['policy_nonlinear'].__name__.lower()))
                torch.nn.init.constant_(l.bias, 0.01)

        for l in self.stopNet:
            if l._get_name() == 'Linear':
                torch.nn.init.xavier_normal_(l.weight, gain=torch.nn.init.calculate_gain(
                    model_config['policy_nonlinear'].__name__.lower()))
                torch.nn.init.constant_(l.bias, 0.01)

        for l in self.baselineNet:
            if l._get_name() == 'Linear':
                torch.nn.init.xavier_normal_(l.weight, gain=torch.nn.init.calculate_gain(
                    model_config['baseline_nonlinear'].__name__.lower()))
                torch.nn.init.constant_(l.bias, 0.01)

    def freeze_agents(self):
        for param in self.filterPolicyNet.parameters():
            param.requires_grad = False
        for param in self.baselineNet.parameters():
            param.requires_grad = False

    def freeze_stopNet(self):
        for param in self.stopNet.parameters():
            param.requires_grad = False

    def unfreeze_stopNet(self):
        for param in self.stopNet.parameters():
            param.requires_grad = True

    def unfreeze_agents(self):
        for param in self.filterPolicyNet.parameters():
            param.requires_grad = True
        for param in self.baselineNet.parameters():
            param.requires_grad = True

    def initLoggers(self):
        """
        Initialize auxiiliary lists to calculate rewards at end of processing
        """

        self.stop_actions = []
        self.filter_actions = []

        ## Log of selected action for filtering
        self.log_pi_filter = []

        ## Baseline values for states
        self.baselines = []

        ##Filter decision specific grad masks (only when their action is taken into account)
        self.filter_grad_masks = []

        ##List of filter statuses
        self.filter_statuses = []

        ##List of predictions at each chekpoint, used for stop decision training
        self.checkpoint_logits = []

        ## Used for calculation of earliness reward
        self.cumsum_filter_statuses = None

    def forward(self, x, filter_status):
        """
        Return stop and filtering decisions of network
        """

        filter_action, log_pi_filter = self.filterPolicy(x, filter_status)
        self.log_pi_filter.append(log_pi_filter)

        stop_action = self.stopNetwork(torch.cat((x.detach(), filter_action.unsqueeze(1).detach()), -1))

        self.stop_actions.append(stop_action)
        stop_action = (stop_action >= self.stop_action_thresh).float()

        b = self.baselineNet(x.detach()).squeeze()
        self.baselines.append(b)

        return stop_action, filter_action

    def filterPolicy(self, x, filter_status):
        """
        Return the filtering action of the policy
        """

        ## Continuous normal distribution
        if self.filter_dist == 'normal':

            filterOut = self.filterPolicyNet(x)
            mu = torch.tanh(filterOut)

            distribution = Normal(mu, self.filter_std)

            filter_float = distribution.sample()
            self.filter_actions.append(filter_float)
            log_pi = distribution.log_prob(filter_float).squeeze()
            filter_float = torch.clamp(filter_float, -1, 1)
            filter_float = torch.abs(filter_float)

            candidate_action_float = (filter_status * filter_float.squeeze(1))
            filter_action = (candidate_action_float[..., None] - self.filter_floats).abs().argmin(-1).long()


        else:
            ## Continuous beta distribution

            filterOut = F.softplus(self.filterPolicyNet(x)) + 1
            alpha, beta = filterOut[:, 0], filterOut[:, 1]

            distribution = Beta(alpha, beta)
            filter_float = distribution.sample()
            self.filter_actions.append(filter_float)
            candidate_action_float = (filter_status * filter_float.detach())
            filter_action = (candidate_action_float[..., None].detach() - self.filter_floats).abs().argmin(-1).long()
            log_pi = distribution.log_prob(filter_float.detach()).squeeze()

        return filter_action, log_pi

    def stopNetwork(self, x):
        """
        Return the stop decision of the network
        """

        ## We assign 1 to stop, 0 to continue

        action = torch.sigmoid(self.stopNet(x)).squeeze()
        return action

    def getRewards(self, logits, labels):
        """
        Calculate the rewards for the filtering and stopping policies
        """
        y_hat = torch.softmax(logits.detach(), dim=1)
        y_hat = torch.max(y_hat, 1)[1]

        MinFilterSum = 1
        MaxFilterSum = 1 + self.num_checkpoints

        MinEarlReward = -1
        MaxEarlReward = 1

        earl_reward = mapValue(MinFilterSum, MaxFilterSum, MaxEarlReward, MinEarlReward, self.cumsum_filter_statuses)

        acc_reward = (2 * (
                y_hat.float().round() == labels.squeeze().float()).float() - 1)

        # Calculate final reward based on earliness and accuracy for filter decisions
        filter_reward = (1 - self.earl_factor) * acc_reward + self.earl_factor * earl_reward
        filter_reward = filter_reward.unsqueeze(1)

        ## Calculate reward for stop decisions
        y_hat_stop = torch.stack(self.checkpoint_logits).detach().transpose(0, 1).softmax(dim=-1).max(-1)[
            1]

        stop_acc_reward = (2 * (y_hat_stop.float().round() == labels.unsqueeze(1).float()) - 1)

        stop_earl_reward = mapValue(MinFilterSum, MaxFilterSum, MaxEarlReward, MinEarlReward,
                                    torch.stack(self.filter_statuses).transpose(0, 1).cumsum(-1))

        stop_reward = (1 - self.earl_factor) * stop_acc_reward + self.earl_factor * stop_earl_reward

        return filter_reward.detach(), stop_reward.detach()

    def discount_rewards(self, rewards, gamma=0.99):
        """
        Discount the policy rewards with given gamma
        """

        rewards = rewards * (gamma ** torch.arange(0, rewards.shape[-1]))

        rewards = torch.flip(torch.cumsum(torch.flip(rewards, (-1,)), -1), (-1,))
        rewards = rewards / (gamma ** torch.arange(0, rewards.shape[-1]))
        return rewards

    def computeLoss(self, logits, labels):
        """
        Calculate loss using REINFORCE algorithm
        """

        ## We skip the last step since the agent cannot decide after last checkpoint, but the last slice log_pi has been added to the list
        ## (for simplicity of implementation)
        log_pi_filter = torch.stack(self.log_pi_filter).transpose(0, 1)[:, :-1]
        baselines = torch.stack(self.baselines).transpose(0, 1)[:, :-1]

        ## Grad mask vector is used because some of the batch samples may have stopped at earlier points
        ## It helps with the correct calculation of the rewards for valid actions
        grad_mask = torch.stack(self.filter_grad_masks).detach().transpose(0,
                                                                           1)[:,
                    :-1]

        R_filter, R_stop = self.getRewards(logits, labels)

        ## Filter reward is adjusted with grad mask
        R_filter = R_filter * grad_mask

        if self.discount:
            R_filter = self.discount_rewards(R_filter)

        b = grad_mask * baselines  # Baseline values of states are also adjusted based on grad mask

        adjusted_rewards = R_filter - b.detach()  # Baseline values are subtracted from achieved rewards

        ## Baseline loss, to train baseline estimation network
        loss_b = F.mse_loss(b,
                            R_filter)

        ## If reward at any checkpoint is higher than future rewards, it means agent should stop
        ## The following code implements this
        stop_actions = torch.stack(self.stop_actions).transpose(0,
                                                                1)[:,
                       :-1]
        stop_target = torch.zeros_like(stop_actions)
        ## This adjusts for samples in the batch that have stopped processing, so it makes the reward arbitrarily negative
        ## for correct calculation of the stop reward
        adj_R_stop = torch.where(torch.stack(self.filter_grad_masks).detach().transpose(0,
                                                                                        1) == 1, R_stop,
                                 -5 * torch.ones_like(R_stop))
        for i in range(self.num_checkpoints):
            stop_target[:, i] = (adj_R_stop[:, i:i + 1] > adj_R_stop[:, i + 1:]).all(-1)

        ## Loss for stopping network
        loss_stop = F.binary_cross_entropy(stop_actions.flatten(), stop_target.flatten(), reduction='sum')

        ## Loss for filtering policy (REINFORCE method)
        loss_filter = (-log_pi_filter * adjusted_rewards).sum()

        return loss_stop, loss_filter, loss_b, R_filter.detach().sum(
            -1).mean()
