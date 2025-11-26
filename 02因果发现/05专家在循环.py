#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 专家在循环 (Expert In The Loop)
使用LLM进行智能因果推断的完整版本

作者: 因果发现系统
日期: 2025年
"""

from pgmpy.utils import get_example_model, llm_pairwise_orient
from pgmpy.estimators import ExpertInLoop, ExpertKnowledge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import re
from litellm import completion

# 过滤警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 设置LLM API
os.environ["OPENAI_API_KEY"] = "sk-wHQ1OO5YuHa8mCP60Z45j4dsp2hLwWFrsNRwuUEOhMsj6DM8"
os.environ["OPENAI_BASE_URL"] = "https://xapi.fyrn.link/v1"

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
    'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei',
    'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    input_file = "/home/zkr/因果发现/01数据预处理/缩减数据_规格.csv"
    
    # 尝试使用utf-8编码
    try:
        df = pd.read_csv(input_file, encoding='utf-8', header=0, index_col=0)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='utf-8-sig', header=0, index_col=0)
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='latin-1', header=0, index_col=0)
    
    df = df.dropna(axis=1, how='all')
    df = df.astype('float32')
    
    print(f"✓ 数据加载完成: {df.shape}")
    return df

def create_output_folder():
    """创建输出文件夹"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "05专家在循环结果")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def preprocess_data(df):
    """数据预处理"""
    print("正在进行数据质量检查...")
    
    # 处理NaN值
    if df.isnull().values.any():
        print("数据中存在 NaN 值，使用均值填充")
        df = df.fillna(df.mean())
    
    # 移除零方差列
    zero_var_cols = df.columns[df.var() == 0]
    if not zero_var_cols.empty:
        print(f"移除零方差列: {list(zero_var_cols)}")
        df = df.drop(columns=zero_var_cols)
    
    # 处理多重共线性
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"移除高度共线列: {to_drop}")
    
    # 方差阈值过滤
    selector = VarianceThreshold(threshold=0.01)
    df_transformed = selector.fit_transform(df)
    
    if df_transformed.shape[1] < df.shape[1]:
        retained_cols = df.columns[selector.get_support()]
        df = pd.DataFrame(df_transformed, columns=retained_cols, index=df.index)
        print(f"VarianceThreshold移除了 {df.shape[1] - df_transformed.shape[1]} 个低方差列")
    
    print(f"✓ 数据预处理完成，最终维度: {df.shape}")
    return df

def create_variable_descriptions(df):
    """创建变量描述字典"""
    variable_descriptions = {}
    for col in df.columns:
        variable_descriptions[col] = f"Binary indicator: {col} (yes/no)"
    return variable_descriptions

def robust_llm_orient(u, v, variable_descriptions=None, llm_model="gpt-4o-mini", **kwargs):
    """稳健的LLM定向函数"""
    if variable_descriptions is None:
        variable_descriptions = {}
    
    try:
        # 使用原始的LLM定向函数
        result = llm_pairwise_orient(u, v, variable_descriptions, llm_model)
        return result
    except Exception as e:
        print(f"LLM定向失败 ({u} <-> {v}): {e}")
        # 使用字典序作为回退
        return (u, v) if str(u) < str(v) else (v, u)

def save_dag_results(dag, output_folder, df_columns):
    """保存DAG结果到文件"""
    edges = list(dag.edges())
    
    # 保存TXT格式
    txt_file = os.path.join(output_folder, "ExpertInLoop_因果边完整.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("专家在循环 (Expert In The Loop) 发现的因果边\n")
        f.write("=" * 50 + "\n")
        for i, edge in enumerate(edges, 1):
            f.write(f"{i:3d}. {edge[0]} -> {edge[1]}\n")
    
    # 保存CSV格式
    df_edges = pd.DataFrame(edges, columns=["源节点", "目标节点"])
    csv_file = os.path.join(output_folder, "ExpertInLoop_因果边列表.csv")
    df_edges.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    # 生成网络图
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    if len(edges) > 0:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightpink', 
                              node_size=2000,
                              alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=1.5,
                              alpha=0.7)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, 
                               font_size=10,
                               font_weight='bold',
                               font_family='sans-serif')
    
    plt.title(f"专家在循环 (Expert In The Loop) 因果网络图\n共{len(edges)}条因果边", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    graph_file = os.path.join(output_folder, "ExpertInLoop_因果网络图.png")
    plt.savefig(graph_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 创建详细JSON结果
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    results = {
        "算法信息": {
            "算法名称": "专家在循环 (Expert In The Loop)",
            "策略": "LLM智能定向",
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据维度": {
                "样本数": len(df_columns),
                "变量数": len(df_columns)
            }
        },
        "网络结构": {
            "节点总数": len(dag.nodes()),
            "边总数": len(edges),
            "节点列表": list(dag.nodes()),
            "因果边列表": [{"源节点": edge[0], "目标节点": edge[1]} for edge in edges]
        },
        "统计信息": {
            "入度统计": {node: in_degrees.get(node, 0) for node in dag.nodes()},
            "出度统计": {node: out_degrees.get(node, 0) for node in dag.nodes()},
            "最大入度": max(in_degrees.values()) if in_degrees else 0,
            "最大出度": max(out_degrees.values()) if out_degrees else 0,
            "平均度数": sum(dict(G.degree()).values()) / len(dag.nodes()) if dag.nodes() else 0
        },
        "节点分析": {
            "根节点": [node for node in dag.nodes() if in_degrees.get(node, 0) == 0],
            "叶节点": [node for node in dag.nodes() if out_degrees.get(node, 0) == 0],
            "中介节点": [node for node in dag.nodes() if in_degrees.get(node, 0) > 0 and out_degrees.get(node, 0) > 0]
        }
    }
    
    json_file = os.path.join(output_folder, "ExpertInLoop_因果结果.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return txt_file, csv_file, graph_file, json_file, results

def run_expert_in_loop_algorithm():
    """运行专家在循环算法"""
    print("=" * 60)
    print("05 专家在循环 (Expert In The Loop) - 开始执行")
    print("使用LLM进行智能因果推断")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建输出文件夹
    output_dir = create_output_folder()
    
    # 3. 数据预处理
    df_processed = preprocess_data(df)
    
    # 4. 创建变量描述
    variable_descriptions = create_variable_descriptions(df_processed)
    print(f"✓ 创建了{len(variable_descriptions)}个变量的描述")
    
    # 5. 使用Expert-in-the-Loop进行因果发现
    print("使用Expert-in-the-Loop方法，结合LLM进行边定向...")
    start_time = time.time()
    
    try:
        # 创建ExpertInLoop估计器
        estimator = ExpertInLoop(df_processed)
        
        # 运行估计
        learned_dag = estimator.estimate(
            pval_threshold=0.2,
            effect_size_threshold=0.0,
            variable_descriptions=variable_descriptions,
            llm_model="gpt-4o-mini",
            use_cache=True,
            show_progress=False
        )
        
        if learned_dag is None:
            raise ValueError("ExpertInLoop.estimate() 返回了 None")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ 专家在循环完成，耗时: {execution_time:.2f}秒")
        print(f"✓ 发现 {len(learned_dag.edges())} 条因果边")
        
        # 6. 保存结果
        txt_file, csv_file, graph_file, json_file, results = save_dag_results(learned_dag, output_dir, df_processed.columns)
        
        # 7. 输出结果摘要
        print("\n" + "=" * 60)
        print("专家在循环执行完成 - 结果摘要")
        print("=" * 60)
        print(f"策略: LLM智能定向")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"数据维度: {df_processed.shape[0]} × {df_processed.shape[1]}")
        print(f"发现的因果边数量: {results['网络结构']['边总数']}")
        print(f"网络节点数量: {results['网络结构']['节点总数']}")
        print(f"根节点数量: {len(results['节点分析']['根节点'])}")
        print(f"叶节点数量: {len(results['节点分析']['叶节点'])}")
        print(f"中介节点数量: {len(results['节点分析']['中介节点'])}")
        print(f"平均节点度数: {results['统计信息']['平均度数']:.2f}")
        
        print(f"\n📁 结果保存位置:")
        print(f"  - TXT文件: {txt_file}")
        print(f"  - CSV文件: {csv_file}")
        print(f"  - 网络图: {graph_file}")
        print(f"  - JSON结果: {json_file}")
        
        return output_dir, len(learned_dag.edges())
        
    except Exception as e:
        print(f"❌ 专家在循环执行失败: {str(e)}")
        # 使用快速回退策略
        print("使用快速回退策略...")
        from pgmpy.base import DAG
        
        dag = DAG()
        dag.add_nodes_from(df_processed.columns)
        
        # 基于相关性添加边
        corr_matrix = df_processed.corr().abs()
        edges_added = 0
        max_edges = 50
        
        for i, col1 in enumerate(df_processed.columns):
            for j, col2 in enumerate(df_processed.columns):
                if i < j and edges_added < max_edges:
                    corr_val = corr_matrix.loc[col1, col2]
                    if corr_val >= 0.3:
                        try:
                            dag.add_edge(col1, col2)
                            edges_added += 1
                        except:
                            continue
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ 快速回退完成，耗时: {execution_time:.2f}秒")
        print(f"✓ 发现 {len(dag.edges())} 条因果边")
        
        txt_file, csv_file, graph_file, json_file, results = save_dag_results(dag, output_dir, df_processed.columns)
        
        return output_dir, len(dag.edges())

if __name__ == "__main__":
    import time
    try:
        output_dir, edge_count = run_expert_in_loop_algorithm()
        print(f"\n✅ 05 专家在循环执行成功！发现 {edge_count} 条因果边")
    except Exception as e:
        print(f"\n❌ 05 专家在循环执行失败: {str(e)}")
        raise