import numpy as np
import plotly.graph_objs as go
import torch

def visualize_bboxed_points(
    x, y, z,
    fg_bbox_x, fg_bbox_y, fg_bbox_z,
    bg_bbox_x, bg_bbox_y, bg_bbox_z,
):
    return go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker={
                'size': 0,
                'opacity': 0.6,
            }
        ),
        go.Mesh3d(  # foreground bbox
            x=fg_bbox_x,
            y=fg_bbox_y,
            z=fg_bbox_z,
            i=[0, 2, 0+4, 2+4, 0, 6, 0+1, 6+1, 0, 4, 0+2, 4+2],
            j=[1, 1, 1+4, 1+4, 2, 0, 2+1, 0+1, 1, 5, 1+2, 5+2],
            k=[2, 3, 2+4, 3+4, 6, 4, 6+1, 4+1, 4, 1, 4+2, 1+2],
            color="red",
            opacity=0.3,
            flatshading=True,
        ),
        go.Mesh3d(  # foreground bbox
            x=bg_bbox_x,
            y=bg_bbox_y,
            z=bg_bbox_z,
            i=[0, 2, 0+4, 2+4, 0, 6, 0+1, 6+1, 0, 4, 0+2, 4+2],
            j=[1, 1, 1+4, 1+4, 2, 0, 2+1, 0+1, 1, 5, 1+2, 5+2],
            k=[2, 3, 2+4, 3+4, 6, 4, 6+1, 4+1, 4, 1, 4+2, 1+2],
            color="yellow",
            opacity=0.3,
            flatshading=True,
        ),
    ], layout=go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    ))

    # fig.update_layout(
    #     scene_camera={
    #         "up": {"x": 0, "y": 1, "z": 0},
    #         # "center": {"x": tx+s/2, "y": ty+s/2, "z": tz+s/2},
    #         # "eye": {"x": 10, "y": 10, "z": 10},
    #     }
    # )


def visualize_sdf_grid(
    x, y, z, density, point_size=0, opacity=0.5
):
    return go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker={
                'size': point_size,
                "color": density,
                'opacity': opacity,
                "colorbar": {
                    "thickness": 10,
                }
            }
        ),
    ], layout=go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    ))

def sdf_points_voxelization(
    grid_size: int,
    points: list[np.ndarray],
    min_x, max_x,
    min_y, max_y,
    min_z, max_z,
    FLOAT_TOLERANCE: float = 1e-5,
) -> tuple[np.ndarray, torch.Tensor]:

    V_sdf = torch.ones([1, 1, grid_size, grid_size, grid_size])

    # vsdf_x = np.linspace(fg_min_x, fg_max_x, num=INIT_GRID_SIZE_FG)
    # vsdf_y = np.linspace(fg_min_y, fg_max_y, num=INIT_GRID_SIZE_FG)
    # vsdf_z = np.linspace(fg_min_z, fg_max_z, num=INIT_GRID_SIZE_FG)
    vsdf_grid = np.mgrid[
        min_x:max_x:complex(grid_size),
        min_y:max_y:complex(grid_size),
        min_z:max_z:complex(grid_size),
    ]
    STEP_SIZE = vsdf_grid[0][0][0][0]-vsdf_grid[0][1][0][0]
    ROUNDING_OFS = [
        vsdf_grid[0, :, 0, 0][0],  # x
        vsdf_grid[1, 0, :, 0][0],  # y
        vsdf_grid[2, 0, 0, :][0]   # z
    ]

    
    for point in points:
        px, py, pz = np.round((point-ROUNDING_OFS)/STEP_SIZE)*STEP_SIZE+ROUNDING_OFS
        # print("POS:  ", px, py, pz)

        vx = np.where(np.isclose(vsdf_grid[0, :, 0, 0], px, FLOAT_TOLERANCE))[0][0]
        vy = np.where(np.isclose(vsdf_grid[1, 0, :, 0], py, FLOAT_TOLERANCE))[0][0]
        vz = np.where(np.isclose(vsdf_grid[2, 0, 0, :], pz, FLOAT_TOLERANCE))[0][0]

        # print("GRID: ", vx, vy, vz)
        if V_sdf[0][0][vx, vy, vz] > -1-(-0.01):
            V_sdf[0][0][vx, vy, vz] += -0.01



    return vsdf_grid, V_sdf

