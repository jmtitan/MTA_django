from django.shortcuts import render
from django.template.response import TemplateResponse
import zipfile
import os
from .utils import *

num_progress = 0
data_cfg = load_config('./MTA_app/configs/criteo_cfg.txt')
cfgs = load_config('./MTA_app/configs/DCRMTA.txt')
attr_text_path = 'MTA_app/files/attr/log.txt'

def un_zip(file_name):
   """unzip zip file"""
   zip_file = zipfile.ZipFile(file_name)
   file_path = file_name.replace('origin.zip', '')
   if os.path.isdir(file_path + "pkls"):
       pass
   else:
       os.mkdir(file_path + "pkls")
       for names in zip_file.namelist():
           if '__MACOSX/' in names:
               continue
           zip_file.extract(names, file_path + "pkls")
       zip_file.close()

def SVcal(request):
    global num_progress
    if os.path.isfile('MTA_app/files/attr/log.txt'):
        print('log file exits')
        return JsonResponse(100, safe=False)
    trainer = Trainer(cfgs, data_cfg)
    U, C, T, Y, cost, _, cat1_9 = load_data(data_cfg)
    count = 0
    attr_seq = []
    length = len(C)
    for u, c, f in zip(U, C, cat1_9):
        # if len(c) > 12:
        count += 1
        num_progress = count * 100 / length
        #  计算 Shapely Value
        SV = calculate_shapley(u, c, f, trainer)
        attr = SVattr(SV)
        attr_seq.append(attr)
    text_save(attr_text_path, attr_seq)
    print('finish')
    return JsonResponse(100, safe=False)

def attr(request):
    attr_seq = text_read(attr_text_path)
    U, C, T, Y, cost, _, cat1_9 = load_data(data_cfg)
    # 计算指标
    test_seq = []
    cost_dic = {}
    roi_dic = {}
    for i in range(12):
        cost_dic[f'广告{i + 1}'] = 0
        roi_dic[f'广告{i + 1}'] = 0

    flag = 0  # 标记journey序列, 用于后续黑名单添加处理
    roi = getROI(attr_seq, C, Y, cost)
    for i, r in enumerate(roi):
        roi_dic[f'广告{i + 1}'] = r

    total_Budget = sum([sum(i) for i in cost])
    Budget = total_Budget * cfgs['Budget_proportion']  # 定义budge
    budget_alloc = getBudgetAttr(roi, B=Budget)
    for c, y, cost, t in zip(C, Y, cost, T):
        for ci, costi, ti in zip(c, cost, t):
            test_seq.append([ti, ci, costi, y / len(c), flag])
            cost_dic[f'广告{ci + 1}'] += costi
        flag += 1
    test_seq.sort(key=lambda x: x[0], reverse=False)
    conv_num, tot_cost, c_conv = attr_criterion(budget_alloc, test_seq)

    Y_conv_num = sum(Y)
    conv_dic = {'原转化量':Y_conv_num, '重分配后转化量':conv_num}
    cost_dic = {'原成本':total_Budget, '重分配后成本':tot_cost}
    gain_dic = {'原始分配':Y_conv_num/total_Budget, '重分配后':conv_num/tot_cost}
    return render(request, 'MTA_app/attr.html',
                  {'roi': roi_dic.items(), 'c_conv': c_conv.items(), 'convert': conv_dic.items(),
                   'cost': cost_dic.items(), 'gain':gain_dic.items()})


def stat(request):
    un_zip('./MTA_app/files/origin.zip')
    data_cfg = load_config('./MTA_app/configs/criteo_cfg.txt')
    U, C, T, Y, cost, CPO, cat1_9 = load_data(data_cfg)
    tot_campaigns = sum([len(i) for i in C])
    cost_dict = {}
    c_dict = {}
    for i in range(12):
        c_dict[f'广告{i+1}'] = 0
        cost_dict[f'广告{i+1}'] = 0
    for ci, costi in zip(C,cost):
        for cij, costij in zip(ci,costi):
            c_dict[f'广告{cij+1}'] += 1
            cost_dict[f'广告{cij+1}'] = costij

    return render(request, 'MTA_app/stat.html',
                  {'tot_campaigns':tot_campaigns, 'c_dict':c_dict.items(), 'cost_dict':cost_dict.items()})

def res_process(request):
    print('show_progress----------'+str(num_progress))
    return JsonResponse(num_progress, safe=False)


def wait(request):
    # return JsonResponse(num_progress, safe=False)
    return render(request, 'MTA_app/wait.html')

def test(request):
    return TemplateResponse(request, 'MTA_app/wait.html')

