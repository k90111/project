import numpy as np
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from collections import Counter
# ----------------------------------------------------------------------------------------
# 方法区
# 读取 CSV 文件到 DataFrame
def duqu():
    df = pd.read_csv(r'D:\python\.idea\2024.2.5\pythonProject--\data.csv')

# 环境和参数的配置--使表格美 工具函数set_seaborn_properties
# 设置 Seaborn 的全局样式和参数。
# context默认大小值  font--华文细黑  图像12*8英寸 清晰度150 避免乱码-去除负号
def set_seaborn_properties(context='talk', font_scale=0.8):
    sns.set_theme(context=context, font='STXIHEI', font_scale=font_scale,
                  rc={'axes.unicode_minus': False,
                      'figure.figsize': (12, 8),
                      'figure.dpi': 150})
# ----------------------------------------------------------------------------------------------------
# # 工具函数 get_2025_sleep_dataframe 获取数据集并且封装到pandas里面，分组将人的ID作为索引
# def get_2025_sleep_dataframe():
#     # 标准化列名（去除前后空格和特殊字符）
#     df.columns = df.columns.str.strip()
#     # 按 'Person' 列分组（每个ID为一组）
#     PersonID_group = df.groupby('Person ID')
#     # 初始化一个空 DataFrame 用于存储结果
#     PersonID_2025_df = pd.DataFrame()
#      # 返回结果，并将 'Person' 列设为索引
#     return PersonID_2025_df.set_index('Person ID')
# ----------------------------------------------------------------------------------------------------
# 设置索引
def index():
    # 加载数据
    df = pd.read_csv(r'D:\python\.idea\2024.2.5\pythonProject--\data.csv')
    # 标准化列名（去除前后空格和特殊字符）
    df.columns = df.columns.str.strip()
    # 确认列名
    print("当前列名:", df.columns.tolist())
    # 设置索引
    try:
        df = df.set_index('Person ID')
        print("索引设置成功！")
        print(df.head())
    except KeyError as e:
        print("设置索引失败:", e)
        print("可用列名:", df.columns.tolist())
 # ----------------------------------------------------------------------------------------------------
# 请理数据
def clean():
    df = pd.read_csv(r'D:\python\.idea\2024.2.5\pythonProject--\data.csv')
    # 查看前几行
    df.head();
    # 检查缺失值数量
    df.isnull().sum();
    # 删除缺失值
    df=df.dropna();
    # 处理重复值
    df=df.drop_duplicates();
# ----------------------------------------------------------------------------------------------------
#对研究对象绘制睡眠时长直方图分析
def sleeptime_anlysis():
    # 清楚缓存
    plt.close('all')
    # 设置表的格式
    set_seaborn_properties()
    plt.subplots_adjust(hspace=1, wspace=0.7)
    # 绘制睡眠时长直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df['Sleep Duration'], bins=20, color='lightgreen', edgecolor='black')
    plt.title('Sleep Duration Distribution')
    plt.xlabel('Sleep Duration (hours)')
    plt.ylabel('Count')
    plt.grid(True)
    # 保存的显示
    plt.savefig('sleep.png', facecolor='white', transparent=False)
    plt.show()
# ----------------------------------------------------------------------------------------------------
def occupation_counts():
    # 统计职业出现的频率
    occupation_counts = Counter(df['Occupation'])
    # 创建词云对象
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',  # 颜色方案
        max_words=50,  # 最多显示50个词
        max_font_size=150  # 最大字体大小
    ).generate_from_frequencies(occupation_counts)
    plt.close('all')
    # 绘制词云图
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 隐藏坐标轴
    plt.title('职业分布词云图', fontsize=20, pad=20)
    plt.show()
    wordcloud.to_file('occupation_wordcloud.png')
# ----------------------------------------------------------------------------------------------------
# 对年龄为主轴对睡眠时长和睡眠质量的线性关系和研究
# 按年龄分组计算均值
def age_sleeptime_quality():
    # 取列名分组 设置分组的大小
    age_group = df.groupby('Age')[['Sleep Duration', 'Quality of Sleep']].mean().reset_index()
    plt.figure(figsize=(14, 7))
    # 双Y轴折线图绘制
    # 睡眠时长（主坐标轴）
    ax = sns.lineplot(
        x='Age',
        y='Sleep Duration',
        data=age_group,
        color='#1f77b4',
        marker='o',
        markersize=4,
        label='Sleep Duration'
    )
    # 展示平均睡眠时长（单位：小时，蓝色折线  # 1f77b4） 固定y轴范围（5 - 9小时），便于跨图表比较
    plt.ylabel('Hours', color='#1f77b4')
    plt.ylim(5, 9)  # 固定y轴范围便于比较
    # 睡眠质量（次坐标轴）
    ax2 = plt.twinx()
    sns.lineplot(
        x='Age',
        y='Quality of Sleep',
        data=age_group,
        color='#ff7f0e',
        marker='s',
        markersize=4,
        label='Sleep Quality',
        ax=ax2
    )
    # 展示平均睡眠质量评分（单位：分，橙色折线  # ff7f0e） 固定y轴范围（4 - 9分）
    ax2.set_ylabel('Score (1-10)', color='#ff7f0e')
    ax2.set_ylim(4, 9)
    # 高级美化
    plt.title('Average Sleep Patterns by Age', fontsize=12, pad=25)
    # 使用sns.lineplot绘制平滑折线
    sns.despine()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # 年龄显示为整数
    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.2),
              ncol=2)
    # 保存为高清PNG（dpi = 300，透明背景） 显示交互式图表（plt.show()
    plt.savefig('sleep_by_age.png', dpi=300, transparent=True)
    plt.show()
# ----------------------------------------------------------------------------------------------------
# 以睡眠质量为主轴分析与身体活动水平和压力水平的线性关系
def quality_Stress_PhysicalActivity():
    quality_group = df.groupby('Quality of Sleep')[['Physical Activity Level', 'Stress Level']].mean().reset_index()
    plt.figure(figsize=(12, 7))
    # 身体活动水平（主坐标轴）
    ax = sns.lineplot(
        x='Quality of Sleep',
        y='Physical Activity Level',
        data=quality_group,
        color='#1f77b4',
        marker='o',
        markersize=4,
        label='Physical Activity Level'
    )
    ax.set_xlabel('Quality of Sleep (1-10)', fontsize=12)
    ax.set_ylabel('Physical Activity Level (1-10)', color='#1f77b4', fontsize=12)
    ax.set_ylim(0, 10)  # 调整为1-10的评分范围
    # 压力水平（次坐标轴）
    ax2 = plt.twinx()
    sns.lineplot(
        x='Quality of Sleep',
        y='Stress Level',
        data=quality_group,
        color='#ff7f0e',
        marker='s',
        markersize=4,
        label='Stress Level',
        ax=ax2
    )
    ax2.set_ylabel('Stress Level (1-10)', color='#ff7f0e', fontsize=12)
    ax2.set_ylim(0, 10)  # 调整为1-10的评分范围
    # 高级美化 # 图表美化
    plt.title('quality_Stress_PhysicalActivity', fontsize=15, pad=20)
    sns.despine()
    ax.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2,fontsize=10)
    plt.savefig('PhysicalActivity_stress_quality.png', dpi=300, transparent=True)
    plt.show()
# ----------------------------------------------------------------------------------------------------
# 绘制职业+睡眠时长，质量，压力，活动水平多个维度的差异的平行坐标图
def occuption_sleep_stress_quality_Activity():
    # 数据预处理：按职业计算均值并排序（按睡眠质量降序）
    occupation_stats = df.groupby('Occupation')[
        ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
    ].mean().sort_values('Quality of Sleep', ascending=False).reset_index()
    # 将数据从宽格式转为长格式（便于分面绘制）
    melted_df = pd.melt(
        occupation_stats,
        id_vars='Occupation',
        value_vars=['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level'],
        var_name='Metric',
        value_name='Score'
    )
    # 创建分面点图
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    # 绘制点图 + 连线（突出职业在各指标的位置）
    sns.pointplot(
        data=melted_df,
        x='Score',
        y='Occupation',
        hue='Metric',
        palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # 自定义颜色
        join=False,  # 不连接点
        dodge=True,  # 分开展示不同指标
        markers=['o', 's', 'D', '^'],  # 不同形状标记
        scale=1.2  # 点大小
    )
    # 高级美化
    plt.title('Comparison of multidimensional indicators of occupational and sleep health', pad=20)
    plt.xlabel('Score/hour')
    plt.ylabel('')
    plt.legend(title='index', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine(left=True)
    plt.tight_layout()
    # 保存图片
    plt.savefig('occupation_sleep.png', dpi=300, bbox_inches='tight')
    plt.show()
# ----------------------------------------------------------------------------------------------------
# 用模型,综合数据，用活动水平和压力水平，睡眠时间，步频分类高睡眠人群和低睡眠人群
# 分析健康方式对与睡眠的重要性，并分析健康方式对睡眠质量的边缘效应
# 边缘效应：如果曲线斜率越大，说明边际效应越强。
# 斜线：当某个健康行为发生微小变化时，睡眠质量随之变化的程度。
def yuce():
    # 准备数据
    X = df[['Physical Activity Level', 'Stress Level', 'Sleep Duration', 'Heart Rate']]
    y = df['Quality of Sleep']
    # 二分类转换（假设>7分为高质量睡眠）
    df['High Quality Sleep'] = (df['Quality of Sleep'] > 7).astype(int)
    # 训练分类模型
    clf = RandomForestClassifier()
    clf.fit(X, df['High Quality Sleep'])
    # 绘制特征重要性
    plt.figure(figsize=(10, 5))
    pd.Series(clf.feature_importances_, index=X.columns).sort_values().plot.barh()
    plt.title('Inportance of health way for high quality sleep')
    plt.savefig('Inportance of health way for high quality sleep.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    # 绘制部分依赖图（PDP）
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(
        clf, X, ['Physical Activity Level', 'Stress Level'],
        ax=ax, contour_kw={'cmap': 'coolwarm'}
    )
    plt.suptitle('Marginal effects of health behaviors on sleep quality')
    plt.savefig('Marginal effects of health behaviors on sleep quality.png', dpi=300, bbox_inches='tight')
    plt.show()
#-----------------------------------1----------------------------------------------------
# main运行区
df = pd.read_csv(r'D:\python\.idea\2024.2.5\pythonProject--\data.csv')
duqu()
# 设置索引和清洗
index()
clean()
# 对数据集进行分析
sleeptime_anlysis()
occupation_counts()
age_sleeptime_quality()
quality_Stress_PhysicalActivity()
occuption_sleep_stress_quality_Activity()
yuce()

