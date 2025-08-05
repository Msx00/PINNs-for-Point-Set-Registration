import plotly.graph_objects as go
import numpy as np

def plot_open3d(source, target, result):
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    result_points = np.asarray(result.points)
    print(0)
    # 创建 Plotly 图形
    fig = go.Figure()
    print(1)
    # 添加源点云 (红色)
    fig.add_trace(go.Scatter3d(
        x=source_points[:,0], y=source_points[:,1], z=source_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='red'
        )
    ))
    # 添加目标点云 (绿色)
    fig.add_trace(go.Scatter3d(
        x=target_points[:,0], y=target_points[:,1], z=target_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='green'
        )
    ))
    # 添加配准后的点云 (蓝色)
    fig.add_trace(go.Scatter3d(
        x=result_points[:,0], y=result_points[:,1], z=result_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue'
        )
    ))
    # 显示图形
    fig.show()

def plot_array(source, target, result):
    source_points = source
    target_points = target
    result_points = result
    print(0)
    # 创建 Plotly 图形
    fig = go.Figure()
    print(1)
    # 添加源点云 (红色)
    fig.add_trace(go.Scatter3d(
        x=source_points[:,0], y=source_points[:,1], z=source_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='red'
        )
    ))
    # 添加目标点云 (绿色)
    fig.add_trace(go.Scatter3d(
        x=target_points[:,0], y=target_points[:,1], z=target_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='green'
        )
    ))
    # 添加配准后的点云 (蓝色)
    fig.add_trace(go.Scatter3d(
        x=result_points[:,0], y=result_points[:,1], z=result_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue'
        )
    ))
    # 显示图形
    fig.show()