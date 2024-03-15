# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:52:13 2023

@author: 21714
"""

#!pip install pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 创建一个贝叶斯网络模型
model = BayesianNetwork([('锻炼', '胃痛'), ('饮食', '胃痛'), ('饮食', '腹胀'), ('胃痛', '恶心'), ('胃痛', '胃炎'), ('腹胀', '胃炎')])

# 定义每个节点的条件概率分布 (CPD)
cpd_dl = TabularCPD(variable='锻炼', variable_card=2, values=[[0.5], [0.5]])
cpd_ys = TabularCPD(variable='饮食', variable_card=2, values=[[0.4], [0.6]])
cpd_wt = TabularCPD(variable='胃痛', variable_card=2, values=[[0.2, 0.45, 0.55, 0.7], [0.8, 0.55, 0.45, 0.3]],
                  evidence=['锻炼', '饮食'], evidence_card=[2, 2])
cpd_fz = TabularCPD(variable='腹胀', variable_card=2, values=[[0.2, 0.6], [0.8, 0.4]], evidence=['饮食'], evidence_card=[2])
cpd_ex = TabularCPD(variable='恶心', variable_card=2, values=[[0.7, 0.2], [0.3, 0.8]], evidence=['胃痛'], evidence_card=[2])
cpd_wy = TabularCPD(variable='胃炎', variable_card=2, values=[[0.8, 0.6, 0.4, 0.1], [0.2, 0.4, 0.6, 0.9]],
                  evidence=['胃痛', '腹胀'], evidence_card=[2, 2])

# 将CPD添加到模型
model.add_cpds(cpd_dl, cpd_ys, cpd_wt, cpd_fz, cpd_ex, cpd_wy)

# 验证模型结构和CPD
assert model.check_model()

# 创建推理对象
inference = VariableElimination(model)

# 查询边缘概率
result = inference.query(variables=['胃痛'])
print(result)

# 查询条件概率
result = inference.query(variables=['胃痛'], evidence={'恶心': 1})
print(result)

