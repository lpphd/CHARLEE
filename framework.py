from tsai_models import InceptionTimeClassifier, ResNetClassifier
from rl_policies import *


class CHARLEE(nn.Module):
    def __init__(self, model_config, data_config):
        super(CHARLEE, self).__init__()

        self.dev = model_config['device']
        self.Policy = RLPolicies(model_config, data_config)
        ## Dimensions of hidden state is determined by number of channels, filters per channel and features per filter
        ## In addition, as described in the paper, a float indicates the point in time of processing and the history of actions
        self.hidden_state_dim = 1 * (data_config['channels'] * model_config['num_filters_per_channel'] * model_config[
            'num_feats_per_filter']) + 1 + model_config[
                                    'num_checkpoints'] + 1

        self.ConvNet = torch.nn.Conv1d(in_channels=data_config['channels'],
                                       out_channels=data_config['channels'] * model_config[
                                           'num_filters_per_channel'], kernel_size=model_config['kernel_size'],
                                       groups=data_config['channels'])

        if model_config['use_dl_model'] == 'inceptiontime':
            self.Classifier = InceptionTimeClassifier(model_config, data_config)
        elif model_config['use_dl_model'] == 'resnet':
            self.Classifier = ResNetClassifier(model_config, data_config)
        else:
            print("Invalid option for DL model.")
            exit(-1)

        self.model_config = model_config
        self.data_config = data_config
        self.slice_length = self.data_config['timesteps'] // (self.model_config['num_checkpoints'] + 1)

        ## The predetermined floats that indicate the percentages of the channels that can be kept
        self.filter_floats = torch.linspace(0, 1, self.model_config['num_channel_slices'] + 1)

        ##Not used after all
        self.warmup_epochs = model_config['warmup_epochs']
        self.cooldown_epochs = model_config['cooldown_epochs']
        self.warmup_phase = False
        self.cooldown_phase = False

        self.total_epochs = model_config['n_epochs']
        self.earl_factor = model_config['earliness_weight_factor']

        ## Initial values of numbers/vectors that will be used to calculate the convolution features for the hidden state
        self.ppv_elem_count = None
        self.batch_conv_sum = None
        self.pos_elem_sum = None
        self.pos_index_sum = None
        self.pos_index_start = 0
        self.ppv_pos_count = None

    def create_input_mask(self, input, slice_index, filter_state_vector):
        """
        Given the actions of the agent, appropriately zero-out parts of input that have been filtered-out,
        to prepare data for the final classifier
        """
        mask = torch.zeros_like(input)
        for i in range(slice_index + 1):
            start_point = i * self.slice_length
            end_point = (i + 1) * self.slice_length
            if i == self.model_config['num_checkpoints']:
                end_point = self.data_config['timesteps']
            for j, filter_status in enumerate(filter_state_vector[:, i]):
                mask[j, :torch.ceil(filter_state_vector[j, i] * self.data_config['channels']).int(),
                start_point:end_point] = 1
        return mask.type(torch.bool)

    def create_slice_mask(self, input_slice, filter_status):
        """
        Given the actions of the agent for the specific time slice, appropriately zero-out parts of input that have been filtered-out,
        to prepare data for calculation of the hidden state
        """
        mask = torch.zeros_like(input_slice)
        for j, status in enumerate(filter_status):
            mask[j, :torch.ceil(status * self.data_config['channels']).int(), :] = 1
        return mask.type(torch.bool)

    def action_to_filter_status(self, action):
        """
        Translate agent action (an integer) to appropriate float from predetermined set
        These are equivalent ways of representing the percentage of channels selected
        E.g. if the actions are 0,1,2, 1 corresponds to keeping half the channels, so it would be 0.5 as a float
        """
        return torch.take(self.filter_floats, action)

    def update_hidden_state(self, x, h, filter_status, start_point, end_point):
        """
        Update the hidden state, given input and previous (or initial) hidden state
        """
        ##Filter status should have shape (Batch_size, 1) and be float (either -1 for undetermined
        ## or 0 for stop, 0.2 for 1/5 of channels etc)
        ## Create a mask, first by checking which samples have completely stopped
        ## (so do not updated hidden state at all) and for the ones that continue, only
        ## partially update hidden state based on the filtering percentage

        ## Create slice mask based on agent decision for specific slice
        slice_mask = self.create_slice_mask(x, filter_status)

        ## Mask out input given the created slice mask
        x = x * slice_mask + (
                self.data_config['mask_value'] * torch.ones_like(x) * (~slice_mask))

        ## The "overlap slice" is used to correctly calculate the convolution at the edge of slices
        x = torch.cat((self.overlap_slice, x), dim=-1)
        self.overlap_slice = x[:, :, -(self.model_config['kernel_size'] - 1):]

        ## Perform convolution with masked input and start calculating features
        conv_out = self.ConvNet(x)  # Shape batch_size*out_channels*timesteps
        conv_max = conv_out.max(-1)[0]  # Shape batch_size*out_channels
        conv_min = conv_out.min(-1)[0]

        self.batch_conv_sum = self.batch_conv_sum + conv_out.sum(-1)

        self.ppv_elem_count += conv_out.shape[-1]

        self.ppv_pos_count = self.ppv_pos_count + (conv_out > 0).sum(-1)

        self.pos_elem_sum = self.pos_elem_sum + ((conv_out > 0) * conv_out).sum(-1)

        self.pos_index_sum = self.pos_index_sum + torch.where(conv_out > 0, torch.arange(start_point, end_point),
                                                              torch.zeros_like(conv_out)).sum(-1)

        ##Update hidden state features given the newly calculated ones
        new_max = torch.maximum(conv_max, h[:, :, 0])
        new_min = torch.minimum(conv_min, h[:, :, 1])
        new_ppv = self.ppv_pos_count / self.ppv_elem_count
        new_mean = self.batch_conv_sum / self.ppv_elem_count
        new_mean_pos_sum = torch.where(self.ppv_pos_count > 0, self.pos_elem_sum / (self.ppv_pos_count + 1),
                                       torch.zeros_like(self.ppv_pos_count))
        new_mipv = self.pos_index_sum / self.ppv_elem_count
        update = torch.stack((new_max, new_min, new_ppv, new_mean, new_mean_pos_sum, new_mipv), -1)
        return update

    ## Functions to freeze/unfreeze parts of the network
    def freeze_hs_encoder(self):
        for param in self.ConvNet.parameters():
            param.requires_grad = False

    def unfreeze_hs_encoder(self):
        for param in self.ConvNet.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        for param in self.Classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.Classifier.parameters():
            param.requires_grad = True

    def forward(self, X, test=False, epoch=0):

        if epoch < self.warmup_epochs:
            self.Policy.freeze_agents()
            self.Policy.freeze_stopNet()
            self.freeze_hs_encoder()
            self.warmup_phase = True
        elif self.warmup_epochs <= epoch <= (self.total_epochs - self.cooldown_epochs):
            self.Policy.unfreeze_agents()
            self.Policy.unfreeze_stopNet()
            self.unfreeze_hs_encoder()
            self.warmup_phase = False
            epoch -= self.warmup_epochs
        else:
            self.Policy.freeze_agents()
            self.Policy.freeze_stopNet()
            self.freeze_hs_encoder()
            self.unfreeze_classifier()
            self.warmup_phase = False
            self.cooldown_phase = True

        self.Policy.initLoggers()

        # Initialize hidden state as 0s
        h = torch.zeros(X.shape[0], self.data_config['channels'] *
                        self.model_config['num_filters_per_channel'],
                        self.model_config['num_feats_per_filter'])

        ## Initialize overlap slice to correctly calculate convolution at edges
        self.overlap_slice = torch.zeros_like(X[:, :, -(self.model_config['kernel_size'] - 1):])

        self.batch_conv_sum = torch.zeros(X.shape[0],
                                          self.data_config['channels'] *
                                          self.model_config['num_filters_per_channel'])
        self.pos_elem_sum = torch.zeros_like(self.batch_conv_sum)
        self.ppv_elem_count = 0
        self.pos_index_sum = 0
        self.ppv_pos_count = torch.zeros_like(h[..., 0])

        ## Predictions vector
        predictions = -torch.ones((X.shape[0], self.data_config['n_classes']))

        ## Filter status (as float)
        filter_status = torch.ones((X.shape[0]), requires_grad=False)

        ## Filter status for all checkpoints (as float)
        filter_state_vector = torch.zeros((X.shape[0], self.model_config['num_checkpoints'] + 1), requires_grad=False)

        self.cumsum_filter_statuses = -torch.ones((X.shape[0]), requires_grad=False)

        self.full_logits = None
        filter_statuses = []

        for slice_i in range(self.model_config['num_checkpoints'] + 1):

            ## Update filter status in StopPolicy
            self.Policy.filter_statuses.append(filter_status)
            filter_statuses.append(filter_status)

            ## Update vectors
            filter_state_vector[:, slice_i] = filter_status

            start_point = slice_i * self.slice_length
            end_point = (slice_i + 1) * self.slice_length
            if slice_i == self.model_config['num_checkpoints']:
                ##Here we should add the remainder of the timesteps but another slice lenth will cover the end of the series anyway
                end_point = self.data_config['timesteps']

            ## Calculate updated hidden state
            h = self.update_hidden_state(X[:, :, start_point:end_point], h,
                                         filter_status, start_point, end_point)

            ## Calculate input to networks
            h_policy_input = torch.cat(
                (h.flatten(start_dim=1, end_dim=-1),
                 filter_state_vector,
                 torch.ones_like(filter_status.unsqueeze(-1)) * (slice_i / self.model_config['num_checkpoints'])),
                -1)

            ## Input to policy network, with hidden state detached to stop gradient flow backwards
            mask = self.create_input_mask(X[:, :, :end_point], slice_i, filter_state_vector).detach()

            h_classifier_input = X[:, :, :end_point] * mask + (
                    self.data_config['mask_value'] * torch.ones_like(X[:, :, :end_point])) * (~mask)

            logits = self.Classifier(h_classifier_input)
            self.Policy.checkpoint_logits.append(logits)

            stop_action, filter_action = self.Policy(h_policy_input, filter_status)

            ## If filter status is zero, the sample processing has stopped, so the gradient mask is set to 0
            ## In that way only the valid agent decisions (gradients) are taken into account later
            filter_grad_mask = torch.where(filter_status == 0.0,
                                           torch.zeros_like(filter_status), torch.ones_like(filter_status)).detach()

            self.Policy.filter_grad_masks.append(filter_grad_mask)

            ## A stop action of 1 means stop, 0 means continue
            # Filter action is interpreted as is, from 0-1
            if test:
                ## Stop action is taken into account only during testing
                stopped_samples_mask = (filter_action.unsqueeze(1) == 0) | (stop_action.unsqueeze(1) == 1)
            else:
                ## During training, only filtering can stop the sample processing
                stopped_samples_mask = (filter_action.unsqueeze(1) == 0)

            predictions = torch.where(stopped_samples_mask & (predictions == -1), logits, predictions)

            self.cumsum_filter_statuses = torch.where(
                (stopped_samples_mask.squeeze(1) == 1) & (self.cumsum_filter_statuses == -1),
                filter_state_vector.sum(-1), self.cumsum_filter_statuses)

            ## Update the filter status appropriately
            if test:
                filter_status = torch.where((filter_status != 0) & (stop_action == 1),
                                            torch.zeros_like(filter_status),
                                            filter_status)

                filter_status = torch.where((filter_status != 0),
                                            self.action_to_filter_status(filter_action),
                                            filter_status)

            else:
                filter_status = torch.where((filter_status != 0),
                                            self.action_to_filter_status(filter_action),
                                            filter_status)

        predictions = torch.where(predictions == -1, logits, predictions).squeeze()

        self.full_logits = self.Classifier(X)

        self.cumsum_filter_statuses = torch.where(self.cumsum_filter_statuses == -1,
                                                  filter_state_vector.sum(-1),
                                                  self.cumsum_filter_statuses)

        self.Policy.cumsum_filter_statuses = self.cumsum_filter_statuses
        ## Pad filter history in case of early exiting in all samples of batch
        full_filter_history = F.pad(torch.stack(filter_statuses).transpose(0, 1),
                                    [0, self.model_config['num_checkpoints'] - slice_i])
        return predictions, full_filter_history

    def computeLoss(self, logits, labels):
        Lstop, Lfilter, Lbaseline, Rewards = self.Policy.computeLoss(logits, labels)

        y_hat = torch.softmax(logits.detach(), dim=1)
        y_hat = torch.max(y_hat, 1)[1]
        wrong_samples = (y_hat.float().round() != labels.squeeze().float())

        if self.warmup_phase:
            Lacc_main = F.cross_entropy(self.full_logits, labels)
        else:
            Lacc_main = F.cross_entropy(logits, labels)

            if wrong_samples.any():
                Lacc_main += F.cross_entropy(self.full_logits[wrong_samples], labels[wrong_samples])

        Lacc = Lacc_main

        Lacc = self.model_config['lacc_mult'] * Lacc

        return Lacc, Lstop, Lfilter, Lbaseline, Rewards
