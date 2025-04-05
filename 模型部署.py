import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文和负号，这里以微软雅黑为例
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 加载模型
model_path = "LGBMRegressor.pkl"
model = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="轻量级梯度提升回归模型预测与 SHAP 可视化", page_icon="💕👩‍⚕️🏥")
st.title("💕👩‍⚕️🏥 轻量级梯度提升回归模型预测与 SHAP 可视化")
st.write("通过输入所有变量的值进行单个样本分娩心理创伤的风险预测，可以得到该样本罹患分娩心理创伤的概率，并结合 SHAP 瀑布图分析结果，有助于临床医护人员了解具体的风险因素和保护因素。")

# 特征范围定义
feature_ranges = {
    "年龄": {"type": "numerical", "min": 18, "max": 42, "default": 18},
    "体重": {"type": "numerical", "min": 52, "max": 91, "default": 52},
    "居住地": {"type": "categorical", "options": ["农村", "城市"]},
    "婚姻状况": {"type": "categorical", "options": ["未婚", "已婚"]},
    "就业情况": {"type": "categorical", "options": ["无固定工作", "有固定工作"]},
    "学历": {"type": "categorical", "options": ["初中及以下", "高中", "专科", "本科及以上"]},
    "医疗费用支付方式": {"type": "categorical", "options": ["自费", "城镇职工基本医疗保险", "城乡居民基本医疗保险"]},
    "怀孕次数": {"type": "numerical", "min": 1, "max": 8, "default": 1},
    "分娩次数": {"type": "numerical", "min": 1, "max": 4, "default": 1},
    "分娩方式": {"type": "categorical", "options": ["剖宫产", "顺产转剖宫产", "顺产"]},
    "不良孕产史": {"type": "categorical", "options": ["否", "是"]},
    "终止妊娠经历": {"type": "categorical", "options": ["否", "是"]},
    "妊娠周数": {"type": "numerical", "min": 29, "max": 44, "default": 29},
    "妊娠合并症": {"type": "categorical", "options": ["否", "是"]},
    "妊娠并发症": {"type": "categorical", "options": ["否", "是"]},
    "喂养方式": {"type": "categorical", "options": ["人工喂养", "纯母乳喂养", "混合喂养"]},
    "新生儿是否有出生缺陷或疾病": {"type": "categorical", "options": ["否", "是"]},
    "家庭人均月收入": {"type": "categorical", "options": ["＜5000", "≥5000"]},
    "使用无痛分娩技术": {"type": "categorical", "options": ["否", "是"]},
    "产时疼痛": {"type": "numerical", "min": 0, "max": 10, "default": 0},
    "产后疼痛": {"type": "numerical", "min": 1, "max": 9, "default": 1},
    "产后照顾婴儿方式": {"type": "categorical", "options": ["自己照顾", "丈夫帮忙", "公婆帮忙", "父母帮忙", "保姆照顾"]},
    "近1月睡眠质量": {"type": "categorical", "options": ["很好", "较好", "较差", "很差"]},
    "近1月夜间睡眠时长": {"type": "numerical", "min": 3, "max": 11, "default": 3},
    "近1月困倦程度": {"type": "categorical", "options": ["无", "＜1次/周", "1~2/周", "≥3次/周"]},
    "孕期体育活动等级": {"type": "categorical", "options": ["基本不锻炼", "小锻炼量", "中等锻炼量", "大锻炼量"]},
    "抑郁": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "焦虑": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "侵入性反刍性沉思": {"type": "numerical", "min": 0, "max": 30, "default": 0},
    "目的性反刍性沉思": {"type": "numerical", "min": 0, "max": 28, "default": 0},
    "心理弹性": {"type": "numerical", "min": 6, "max": 30, "default": 6},
    "家庭支持": {"type": "numerical", "min": 0, "max": 10, "default": 0},
}

# 创建分类特征的映射字典
categorical_mappings = {
    "居住地": {"农村": 0, "城市": 1},
    "婚姻状况": {"未婚": 0, "已婚": 1},
    "就业情况": {"无固定工作": 0, "有固定工作": 1},
    "学历": {"初中及以下": 0, "高中": 1, "专科": 2, "本科及以上": 3},
    "医疗费用支付方式": {"自费": 0, "城镇职工基本医疗保险": 1, "城乡居民基本医疗保险": 2},
    "分娩方式": {"剖宫产": 0, "顺产转剖宫产": 1, "顺产": 2},
    "不良孕产史": {"否": 0, "是": 1},
    "终止妊娠经历": {"否": 0, "是": 1},
    "妊娠合并症": {"否": 0, "是": 1},
    "妊娠并发症": {"否": 0, "是": 1},
    "喂养方式": {"人工喂养": 0, "纯母乳喂养": 1, "混合喂养": 2},
    "新生儿是否有出生缺陷或疾病": {"否": 0, "是": 1},
    "家庭人均月收入": {"＜5000": 0, "≥5000": 1},
    "使用无痛分娩技术": {"否": 0, "是": 1},
    "产后照顾婴儿方式": {"自己照顾": 0, "丈夫帮忙": 1, "公婆帮忙": 2, "父母帮忙": 3, "保姆照顾": 4},
    "近1月睡眠质量": {"很好": 0, "较好": 1, "较差": 2, "很差": 3},
    "近1月困倦程度": {"无": 0, "＜1次/周": 1, "1~2/周": 2, "≥3次/周": 3},
    "孕期体育活动等级": {"基本不锻炼": 0, "小锻炼量": 1, "中等锻炼量": 2, "大锻炼量": 3},
}

# 动态生成输入项
st.sidebar.header("变量输入区域")
st.sidebar.write("请输入变量值：")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
        # 将分类特征的字符串值转换为数值
        if feature in categorical_mappings:
            value = categorical_mappings[feature][value]
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("预测"):
    # 模型预测
    predicted_value = model.predict(features)[0]
    st.write(f"Predicted 分娩心理创伤 score: {predicted_value:.2f}%")

    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 获取基础值和第一个样本的 SHAP 值
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[0]  # 如果是列表，取第一个元素
    else:
        base_value = explainer.expected_value  # 如果是单个值，直接使用

    shap_values_sample = shap_values[0]

    # 创建SHAP瀑布图，确保中文显示
    plt.figure(figsize=(6, 4))  # 设置图形尺寸为6x4英寸
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_sample,
            base_values=base_value,
            data=features[0],
            feature_names=list(feature_ranges.keys())  # 使用 feature_ranges 的键作为特征名称
        ),
        max_display=10  # 限制显示的特征数量
    )

    # 保存SHAP瀑布图并展示
    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=600)
    st.image("shap_waterfall_plot.png")
