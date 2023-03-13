import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_mesh_with_point(func, start_pos, size, num_points, mesh_color, points, point_colors, point_zorders=None):
    # Create the meshgrid
    x = np.linspace(start_pos[0], start_pos[0] + size[0], num_points)
    y = np.linspace(start_pos[1], start_pos[1] + size[0], num_points)
    X, Y = np.meshgrid(x, y)

    # Calculate the values of "func" for each point in the meshgrid
    Z = func(X, Y)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Plot the mesh
    ax.scatter(X, Y, Z, c=Z, cmap=mesh_color, zorder=0)

    # Create a list of 3D points from the points
    points3d = [(x, y, func(x, y)) for x, y in points]

    # Plot the points separately from the mesh points
    for i, point in enumerate(points3d):
        zorder = 1
        if point_zorders is not None:
            if hasattr(point_zorders, '__len__'):
                zorder = point_zorders[i]
            else:
                zorder = point_zorders
        ax.scatter(point[0], point[1], point[2], c=point_colors[i], s=50, edgecolors='green', zorder=zorder)
    
    return fig


def plot_3d_points(point_list):
    fig_list = []
    for points in point_list:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        fig_list.append(fig)
    return fig_list


def plot_arrows_3d(x, y, z, dx, dy, dz, ax, color='blue'):
    ax.quiver(x, y, z, dx, dy, dz, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def get_image_list(fig_list):
    image_list = []
    for fig in fig_list:
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:,:,:3]
        image_list.append(image)
    return image_list


def create_animation_from_images(image_list, interval=50):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')

    def animate(i):
        ax.clear()
        ax.imshow(image_list[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(image_list), interval=interval)

    plt.close(fig)

    return anim


def create_and_save_anim(fig_list, filename):
    image_list = get_image_list(fig_list)
    anim = create_animation_from_images(image_list, interval=50)
    anim.save(filename, fps=2, extra_args=['-vcodec', 'libx264'])


def test_plotting_3d():
    n = 5
    arr_size = 10
    point_dim = 3

    arr = np.random.normal(size=(n * arr_size * point_dim))
    arr = np.reshape(arr, newshape=(n, arr_size, point_dim))

    fig_list = plot_3d_points(arr)

    image_list = get_image_list(fig_list)

    anim = create_animation_from_images(image_list, interval=50)

    anim.save('animation.mp4', fps=1, extra_args=['-vcodec', 'libx264'])


if __name__ == "__main__":
    test_plotting_3d()
