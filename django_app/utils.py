import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from django.http import JsonResponse
from django.shortcuts import render
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple
import itertools
from collections import defaultdict
import copy


def load_config(cfg_file):
    f = open(cfg_file, 'r')
    config_lines = f.readlines()
    cfgs = { }
    for line in config_lines:
        ps = [p.strip() for p in line.split('=')]
        if (len(ps) != 2):
            continue
        try:
            if (ps[1].find(',') != -1):
                str_line = ps[1].split(',')
                cfgs[ps[0]] = list(map(int, str_line))
            elif (ps[1].find('.') == -1):
                cfgs[ps[0]] = int(ps[1])
            else:
                cfgs[ps[0]] = float(ps[1])
        except ValueError:
            cfgs[ps[0]] = ps[1]
            if cfgs[ps[0]] == 'False':
                cfgs[ps[0]] = False
            elif cfgs[ps[0]] == 'True':
                cfgs[ps[0]] = True

    return cfgs

def load_data(cfgs):
    file_U = cfgs['file_U']
    file_C = cfgs['file_C']
    file_T = cfgs['file_T']
    file_Y = cfgs['file_Y']
    file_cost = cfgs['file_cost']
    file_CPO = cfgs['file_CPO']
    file_cat1_9 = cfgs['file_cat1_9']

    f = open(file_U, "rb")
    U = pickle.load(f)
    f.close()

    f = open(file_C, "rb")
    C = pickle.load(f)
    f.close()

    f = open(file_T, "rb")
    T = pickle.load(f)
    f.close()

    f = open(file_Y, "rb")
    Y = pickle.load(f)
    f.close()

    f = open(file_cost, "rb")
    cost = pickle.load(f)
    f.close()

    f = open(file_CPO, "rb")
    CPO = pickle.load(f)
    f.close()

    f = open(file_cat1_9, "rb")
    cat1_9 = pickle.load(f)
    f.close()

    return U, C, T, Y, cost, CPO, cat1_9

data_cfg = load_config('./MTA_app/configs/criteo_cfg.txt')
device = torch.device(data_cfg['device'])

class Custom_Dataset(Dataset):
    # dataset for attribution
    def __init__(self, c, u, cat) -> None:
        self.x = list(zip(u, c, cat))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

def pred_collate(batch_data):
    # dataloader fn for attribution
    u = []
    c = []
    cat = []
    # user
    u = torch.LongTensor([i[0] for i in batch_data]).to(device)
    # channel
    C_lens = []
    for jn in batch_data:
        C_lens.append(len(jn[1]))
        c.append(torch.LongTensor(jn[1]))
    c = pad_sequence(c, padding_value=data_cfg['global_campaign_size'], batch_first=True).to(device)
    # feature vector
    cat_padding_list = data_cfg['global_cat_num_list']  # cat1_9 代表 触点的特征数据f
    cat = copy.deepcopy([i[2] for i in batch_data])
    for i in range(len(batch_data)):
        while max(C_lens) > len(cat[i]):
            cat[i].append(cat_padding_list)  # 有些广告的特征向量数与广告数不匹配，则padding
    cat = torch.LongTensor(cat).to(device)
    return [u, c, cat, C_lens]

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRLayer(nn.Module):
    def __init__(self):
        super(GRLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class Data_Emb():
    def __init__(self, config):
        device = torch.device(config['device'])
        self.channel_embedding = nn.Embedding(
            num_embeddings=config['global_campaign_size'] + 1,
            embedding_dim=config['C_embedding_dim']
        ).to(device)

        self.cat_cnt = len(config['cat_embedding_dim_list'])
        self.cat_embedding_list = nn.ModuleList(
            [nn.Embedding(num_embeddings=config['global_cat_num_list'][i] + 1,
                          embedding_dim=config['cat_embedding_dim_list'][i]) for i in range(self.cat_cnt)]
        ).to(device)

        self.user_embedding = nn.Embedding(
            num_embeddings=config['global_user_num'],
            embedding_dim=config['U_embedding_dim']
        ).to(device)

    def __call__(self, U, C, cat):
        embeded_C = self.channel_embedding(C)

        embeded_cat_list = []
        for i in range(self.cat_cnt):
            embeded_cat_list.append(self.cat_embedding_list[i](cat[:, :, i]))  # 特征向量 emb

        concated_tsr = embeded_C  # channel emb
        for i in range(self.cat_cnt):
            concated_tsr = torch.cat((concated_tsr, embeded_cat_list[i]), 2)

        embeded_U = self.user_embedding(U)
        return concated_tsr, embeded_U


class ConvertionPredictor(nn.Module):
    def __init__(self, cfgs, data_cfg):
        super(ConvertionPredictor, self).__init__()
        self.cfgs = cfgs
        self.data_cfg = data_cfg
        self.emb = Data_Emb(data_cfg)
        # self.time_decay = cfgs['time_decay']
        fvec_in = sum(data_cfg['cat_embedding_dim_list'])
        lstm_in = data_cfg['C_embedding_dim'] + fvec_in
        self.num_features = cfgs['predictor_hidden_dim']  # 64
        self.lstm = nn.LSTM(
            input_size=lstm_in,  # 49
            hidden_size=cfgs['predictor_hidden_dim'],  # 64
            num_layers=cfgs['predictor_hidden_layer_depth'],  # 2
            batch_first=True,
            dropout=cfgs['predictor_drop_rate'],
            bidirectional=False
        )

        self.dense_1 = nn.Sequential(
            nn.Linear(cfgs['predictor_hidden_dim'] + lstm_in, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.dense_2 = nn.Sequential(
            nn.Linear(fvec_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
        )
        self.dense_3 = nn.Linear(data_cfg['U_embedding_dim'], 64)
        self.dense_final = nn.Sequential(
            nn.Linear(64, 1)
        )
        self.final_activate = nn.Sigmoid()

        self.dense_rev = nn.Linear(cfgs['predictor_hidden_dim'],
                                   data_cfg['global_campaign_size'] + 1)  # nn.linear 可以兼容不同的维度输入Pytorch linear 多维输入的参数
        if cfgs['gradient_reversal_layer']:  # 是否梯度反转
            self.grl = GRLayer()

    def seq_attention(self, a, s_prev):
        x = torch.cat((a, s_prev), dim=2)
        x = self.dense_1(x)
        # x -= self.time_decay * t
        w = F.softmax(x, dim=1)
        return torch.bmm(torch.transpose(a, 1, 2), w)  # (batch, 64, 1)

    def user_attention(self, f_m, u):
        u = u.unsqueeze(-1)  # (batch, 5, 1)
        # bmm矩阵乘法
        w = torch.bmm(f_m, u)  # 通过内积求相似度
        w = F.softmax(w.squeeze(2), dim=1).unsqueeze(2)  # (batch, num_of_Channels, 1)
        return torch.bmm(torch.transpose(f_m, 1, 2), w).squeeze()  # (batch, 5)

    def BAP(self, features, attentions):
        # Counterfactual Attention Learning
        if len(attentions.shape) < 2:
            attentions = attentions.unsqueeze(0)
        attn_map = attentions.unsqueeze(2)
        B, UF, M = features.size()
        # feature_matrix:(B,M,C) -> (B,M *C)
        feature_matrix = (torch.einsum('bum, bmi->bm', (features, attn_map)) / float(UF)).view(B, -1)
        # sign-sart
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(
            torch.abs(feature_matrix) + 1e-6)  # 12 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attn_map).uniform_(0, 2)  # 零化 + 随机数填充
        else:
            fake_att = torch.ones_like(attn_map)
        counterfactual_feature = (torch.einsum('bum, bmi->bm', (features, fake_att)) / float(UF)).view(B, -1)
        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + 1e-6)
        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)

        return feature_matrix, counterfactual_feature

    def forward(self, U, C, cat, lens):
        tp, u = self.emb(U, C, cat)
        f_vec = tp[:, :, 4:]
        c_rev = None

        packed_input = pack_padded_sequence(  # 压缩填充张量
            input=tp,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )

        # channel-touchpoint feature extraction
        lstm_out, (_, _) = self.lstm(packed_input)
        a, _ = pad_packed_sequence(lstm_out)  # 对压缩填充张量进行解压缩
        a = a.permute(1, 0, 2)  # (batch, channel_seq, hidden_dim)
        if getattr(self, 'grl', None) is not None:
            c_rev = self.dense_rev(self.grl(a))
        c_m = self.seq_attention(a, tp).squeeze()

        # user-touchpoint  feature extraction
        f_m = self.dense_2(f_vec)  # (B, seq, 64)
        u_m = self.dense_3(u)  # (B, 64)
        uf_m = torch.cat((u_m.unsqueeze(dim=1), f_m), dim=1)  # (B, 1+seq, 64)
        uf_attn = self.user_attention(f_m, u_m)  # (B, 64)
        uf_m, uf_m_hat = self.BAP(uf_m, uf_attn)  # (B, 64)

        h_f = uf_m + c_m
        h_f_hat = uf_m_hat + c_m

        p = self.dense_final(h_f)
        p_hat = self.dense_final(h_f_hat)

        return self.final_activate(p), self.final_activate(p - p_hat), c_rev


class Trainer:
    def __init__(self, cfgs, data_cfg):
        self.cfgs = cfgs
        self.data_cfg = data_cfg
        self.batch = cfgs['predictor_batch_size']
        self.model = ConvertionPredictor(cfgs, data_cfg).to(cfgs['device'])
        self.model.load_state_dict(torch.load(cfgs['pretrained_model_path'], map_location=cfgs['device']))


    def predictor(self, c, u, cat):
        with torch.no_grad():
            converts = []
            datald_custom = DataLoader(Custom_Dataset(c, u, cat), batch_size=self.batch, shuffle=False, collate_fn=pred_collate)
            for i, b_data in enumerate(datald_custom):
                u, c, cat, c_lens = b_data
                pred, _, _ = self.model(u, c, cat, c_lens)
                if len(pred) > 1:
                    pred = pred.squeeze()
                else:
                    pred = pred[0]
                converts.extend(pred.cpu().detach().numpy().tolist())
            return converts


def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in itertools.combinations(List, i + 1)]
    return PS

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def v_function(A, C_values):
    '''
    This function computes the worth of each coalition.
    inputs:
            - A : a coalition of channels.
            - C_values : A dictionnary containing the number of conversions that each subset of channels has yielded.
    '''
    subsets_of_A = subsets(A)
    worth_of_A = 0
    for subset in subsets_of_A:
        if subset in C_values:
            worth_of_A += C_values[subset]
    return worth_of_A


def subsets(s):
    '''
    This function returns all the possible subsets of a set of channels.
    input :
            - s: a set of channels.
    '''
    if len(s) == 1:
        return s
    else:
        sub_channels = []
        for i in range(1, len(s) + 1):
            sub_channels.extend(map(list, itertools.combinations(s, i)))
    return sub_channels


def subseq(arr, thred=5):
    finish=[]    # the list containing all the subsequences of the specified sequence
    size = len(arr)    # the number of elements in the specified sequence
    start = 0
    end = 1 << size    # end=2**size
    interv = 1
    if 10 >= size > thred:
        interv = 2**(size - thred)
    if size > 10:
        start = end - 2**10
        interv = 2 **(10 - thred)
    for index in range(start, end, interv):
        array = []    # remember to clear the list before each loop
        for j in range(size):
            if (index >> j) % 2:
                array.append(arr[j])
        # print(array)
        if array:
            finish.append(array)
    return finish

def calculate_shapley(ui, ci, fi, trainer):
    n = len(ci)# no. of channels
    shapley_values = defaultdict(int)
    if n == 1:
        # 避免空集子集
        shapley_values[0] = trainer.predictor([ci], [ui], [fi])[0]
        return shapley_values
    for idx, ci_t in enumerate(ci):
        # u,c,f 除去 ci对应的元素后，计算转化率
        c_tmp = ci.copy()
        f_tmp = fi.copy()
        del c_tmp[idx]
        del f_tmp[idx]
        # 删除元素后，剩余journey的所有子序列（子集太多太慢）
        Sc = subseq(c_tmp)
        Sf = subseq(f_tmp)
        # Sc = subsets(c_tmp)
        # Sf = subsets(f_tmp)
        # 通过矩阵运算方式，获得添加了删除元素的子集集合
        Sc_p = copy.deepcopy(Sc)
        Sf_p = copy.deepcopy(Sf)
        for ele in Sc_p:
            ele.append(ci[idx])
        for ele in Sf_p:
            ele.append(fi[idx])
        # 计算所有子集集合的转化率
        ctf_v = trainer.predictor(Sc, [ui]*len(Sc), Sf)
        # 计算所有添加触点p的子集集合的转化率
        ctf_vp = trainer.predictor(Sc_p, [ui]*len(Sc_p), Sf_p)
        for sc, sv, svp in zip(Sc, ctf_v, ctf_vp):
            weight = (factorial(len(sc)) * factorial(n - len(sc) - 1) / factorial(n))  # Weight = |S|!(n-|S|-1)!/n!
            contrib = svp - sv  # Marginal contribution = v(S U {i})-v(S)
            shapley_values[idx] += weight * contrib
        shapley_values[idx] = shapley_values[idx] if shapley_values[idx] > 0 else 0
    return shapley_values


def SVattr(SV):
    attr = []
    sumSV = sum(list(SV.values()))
    if sumSV == 0:
        attr = [0] * len(SV.keys())
    else:
        for k in SV.keys():
            attr.append(SV[k] / sumSV)
    return attr


def getROI(attr, C, Y, cost, k=12):
    roi = [0] * k
    roi_de = [0] * k
    v = 1
    for a, tp, y, pay in zip(attr, C, Y, cost):
        for i, ci in enumerate(tp):
            roi_de[ci] += pay[i]
            roi[ci] += a[i] * v * y
    for i, de in enumerate(roi_de):
        roi[i] /= de
    return roi


def getBudgetAttr(roi, B):
    summ = sum(roi)
    roi = [i / summ * B for i in roi]
    return roi


def Backeval(cbs, test_seq):
    Blacklist = []
    totcost = 0
    convt_num = 0
    # 统计各个渠道转化数量
    c_conv = {}
    for i in range(12):
        c_conv[f'广告{i+1}'] = 0

    for _, ci, costi, yi, flag in test_seq:
        if flag in Blacklist:
            continue
        else:
            if cbs[ci] > costi:
                totcost += costi
                convt_num += yi
                c_conv[f'广告{ci+1}'] += yi
                cbs[ci] -= costi
            else:
                Blacklist.append(flag)

    return convt_num, totcost, c_conv



def attr_criterion(cbs, test_seq):
    return Backeval(cbs, test_seq)


def text_save(filename, data):
    #filename为写入CSV文件的路径，data为要写入数据列表.
    print(f"存入{filename}...")
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','') #去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功")

def text_read(filename):
    attr = []
    print('读取中...')
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            cont = line.split(' ')
            cont = list(map(float, cont))
            attr.append(cont)
    print(f'{filename}读取完毕')
    return attr




