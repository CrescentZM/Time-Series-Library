import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

# ========================
# TimesBlock：时序块
# ========================
class TimesBlock(nn.Module):
    def __init__(self, configs):  ##configs是为TimesBlock定义的配置
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  ##序列长度
        self.pred_len = configs.pred_len  ##预测长度
        self.k = configs.top_k  ##k表示考虑前k个频率

        # 参数高效设计
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # B: batch size  T: 时间序列的长度  N: 特征数量
        B, T, N = x.size()

        # FFT_for_Period() 将显示，period_list([top_k])表示
        # 前k个显著的周期，period_weight([B, top_k])表示它们的权重（振幅）
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # 填充：为了形成一个二维图，我们需要确保序列的总长度（包括预测部分）能被周期整除
            # 所以需要填充
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # 重塑：我们需要把每个数据的每个通道变成二维变量
            # 并且为了实现后续的2D卷积操作，需要调整维度顺序
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()

            # 2D卷积：从一维变化到二维
            out = self.conv(out)

            # 重塑回来
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # 剔除填充部分并返回结果
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)  # res: 4D [B, length , N, top_k]

        # 自适应聚合
        # 首先，使用softmax从振幅中获取归一化的权重 --> 2D [B, top_k]
        period_weight = F.softmax(period_weight, dim=1)

        # 经过两次unsqueeze(1)，形状变为 [B, 1, 1, top_k]，然后重复权重以适应res的形状
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)

        # 使用加权的top_k结果求和，得到TimesBlock的输出
        res = torch.sum(res * period_weight, -1)

        # 残差连接
        res = res + x
        return res


# ========================
# FFT周期计算
# ========================
def FFT_for_Period(x, k=2):
    # xf形状为[B, T, C]，表示在每个数据点的频率（T），给定批次（B）和特征数（C）
    xf = torch.fft.rfft(x, dim=1)

    # 通过振幅找到周期：在这里我们假设周期性特征在不同批次和通道中是基本常数，
    # 所以我们对这两个维度进行平均，得到频率列表frequency_list，形状为[T]
    # frequency_list中的每个元素表示在频率t上的整体振幅
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 0频率设为0，避免干扰

    # 通过torch.topk()，我们可以得到frequency_list中最大的k个元素，以及它们的位置（即前k主频率）
    _, top_list = torch.topk(frequency_list, k)

    # 返回新的Tensor 'top_list'，它是从当前计算图中分离出来的，且该结果将永远不需要梯度计算。
    # 将其转为numpy格式
    top_list = top_list.detach().cpu().numpy()

    # period：形状为[top_k]，记录各个频率的周期
    period = x.shape[1] // top_list

    # 返回两个值，第2个返回的是一个形状为[B, top_k]的振幅值，表示每个数据的前k个频率的振幅
    return period, abs(xf).mean(-1)[:, top_list]


# ========================
# TimesNet模型
# ========================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # 堆叠 TimesBlock，形成TimesNet的主部分
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # 嵌入层与归一化层
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers  # 编码器层数
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 为不同任务定义的层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        pass

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

