import torch
from torch import nn
from torch.nn.parameter import Parameter


class FeatureWeaver(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        if configs.input_token_len == configs.output_token_len:
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    configs.input_channel,
                    configs.output_channel,
                    5,
                    1,
                    2,
                    1,
                    padding_mode="replicate",
                ),
                nn.LayerNorm([configs.output_channel, configs.seq_len]),
                nn.SiLU(),
                nn.Dropout1d(p=0.1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    configs.output_channel,
                    configs.input_channel,
                    5,
                    1,
                    2,
                    1,
                    padding_mode="replicate",
                ),
                nn.LayerNorm([configs.input_channel, configs.seq_len]),
                nn.SiLU(),
                nn.Dropout1d(p=0.1),
            )
        else:
            self.conv1 = nn.Conv1d(
                configs.input_channel,
                configs.output_channel,
                5,
                1,
                2,
                1,
                padding_mode="replicate",
            )
            self.layernorm_1_x = nn.LayerNorm([configs.output_channel, configs.seq_len])
            self.layernorm_1_y = nn.LayerNorm(
                [
                    configs.output_channel,
                    configs.seq_len
                    - configs.input_token_len
                    + configs.output_token_len,
                ]
            )

            self.conv2 = nn.Conv1d(
                configs.output_channel,
                configs.input_channel,
                5,
                1,
                2,
                1,
                padding_mode="replicate",
            )
            self.layernorm_2_x = nn.LayerNorm([configs.input_channel, configs.seq_len])
            self.layernorm_2_y = nn.LayerNorm(
                [
                    configs.input_channel,
                    configs.seq_len
                    - configs.input_token_len
                    + configs.output_token_len,
                ]
            )

            self.silu = nn.SiLU()
            self.dropout = nn.Dropout1d(p=0.1)

        self.fc = nn.Linear(configs.input_channel, configs.input_channel)

    def forward(self, x, type=0):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # B C L

        if self.configs.input_token_len == self.configs.output_token_len:
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            if type == 0:
                x = self.layernorm_1_x(x)
            else:
                x = self.layernorm_1_y(x)
            x = self.silu(x)
            x = self.dropout(x)

            x = self.conv2(x)
            if type == 0:
                x = self.layernorm_2_x(x)
            else:
                x = self.layernorm_2_y(x)
            x = self.silu(x)
            x = self.dropout(x)

        x = x.permute(0, 2, 1)  # B L C
        x = self.fc(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, ltm):
        super().__init__()
        self.configs = configs
        self.ltm = ltm
        self.feature_weaver = FeatureWeaver(configs)
        self.a = Parameter(torch.ones(1, configs.input_channel))
        self.b = Parameter(torch.ones(1, configs.input_channel))
        self.criterion = nn.MSELoss()
        self.linear = nn.Linear(configs.input_channel, 1)

    def forward(self, batch_x, batch_y=None):
        means = batch_x.mean(1, keepdim=True).detach()
        stdev = batch_x.std(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.where(
            stdev > 1e-2, stdev, torch.tensor(1e-2, device=batch_x.device)
        )

        batch_x = (batch_x - means) / stdev
        x0 = batch_x.permute(0, 2, 1)  # B C L
        x0 = x0.reshape(-1, x0.shape[-1])

        x1 = self.a * batch_x + self.feature_weaver(batch_x, type=0)
        x2 = -self.b * batch_x + self.feature_weaver(batch_x, type=0)
        batch_x = torch.cat([x1, x2], dim=0)
        B = batch_x.shape[0]
        M = batch_x.shape[-1]
        outputs = self.ltm(batch_x)
        # print("outputs shape:", outputs.shape)
        predictions = outputs.reshape(B, -1, outputs.shape[-1])
        predictions = predictions.permute(0, 2, 1)
        y1, y2 = torch.chunk(predictions, 2, dim=0)
        predictions = (y1 - y2) / (self.a + self.b)
        predictions = predictions * stdev + means
        predictions = self.linear(predictions)
        return predictions
