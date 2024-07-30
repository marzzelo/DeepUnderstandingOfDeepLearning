def showAxes(plt, color="gray", linewidth=1, axis_x=True, axis_y=True, grid=False, grid_x=True, grid_y=True):
    ''' 
    Show the x and y axes with gray tiny lines.
     
    Parameters
    ----------
    plt : matplotlib.pyplot
        The pyplot object.
    color : str, optional
        The color of the lines. The default is "gray".
    linewidth : int, optional
        The width of the lines. The default is 1.
    grid : bool, optional
        Whether to show the grid. The default is False.
    '''
    if axis_x:
        plt.axhline(0, color=color, linestyle="-", linewidth=linewidth)

    if axis_y:
        plt.axvline(0, color=color, linestyle="-", linewidth=linewidth)

    if grid:
        if grid_x:
            plt.grid(axis="x", color='lightgray', linestyle="--", linewidth=linewidth)
        if grid_y:
            plt.grid(axis="y", color='lightgray', linestyle="--", linewidth=linewidth)


def showBar(fig, ax, im, size="10%", aspect=20, padding=0.05):  # set color bar height to 1
    """
    Show the color bar of the image.
    parameters:
    fig: the figure object (fig, ax = plt.subplots(figsize=(8, 6)...)
    ax: the axis object
    im: the image object (im = plt.imshow())
    size: the size of the color bar (height)
    aspect: the aspect ratio of the color bar (height/width)
    padding: the padding of the color bar from the image
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Create a new axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=padding)
    # Create the colorbar
    cbar = fig.colorbar(im, cax=cax)
    # Adjust the height of the colorbar
    cax.set_aspect(aspect)  # Adjust the aspect ratio as needed
