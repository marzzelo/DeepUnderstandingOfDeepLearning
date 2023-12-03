def showAxes(plt, color="lightgray", linewidth=1, grid=False):
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
    plt.axhline(0, color=color, linestyle="-", linewidth=linewidth)
    plt.axvline(0, color=color, linestyle="-", linewidth=linewidth)
    if (grid):
        plt.grid(True, color='lightgray', linestyle="--", linewidth=0.5)



if __name__ == "__main__":
    pass