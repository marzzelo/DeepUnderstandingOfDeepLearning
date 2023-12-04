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



if __name__ == "__main__":
    pass
