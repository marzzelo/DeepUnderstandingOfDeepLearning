def box(text, width=60, charset='|' , height=0, wide=False):
    ''' 
    Print a centered box with text, width and char.
    the default width is equal to the text length plus 4.
    the default character set is ['+', '-', '|'].
    '''

    # switch charset
    if charset == '+':
        charset=['+', '-', '|', '+', '+', '+']
    elif charset == '=':
        charset=['╔', '═', '║', '╗', '╚', '╝']
    elif charset == '|':
        charset=['┌', '─', '│', '┐', '└', '┘']
    elif charset == 'B':
        # use block characters
        charset=['█', '█', '█', '█', '█', '█']
    elif charset == 'b':
        charset = ['▄', '▄', '▄', '▄', '▄', '▄']
    else:
        charset = [charset] * 6

    if wide:
        # insert spaces between characters
        text = ' '.join(text)

    # separate the text into lines
    text = text.splitlines()

    # calculate the width as the longest line and the height as the number of lines plus 2
    if width == 0:
        width = max([len(line) for line in text]) + 4

    height = max(height, len(text))

    htop = (height - 2 - len(text)) // 2        # the number of lines on the top
    hbot = (height - 2 - len(text)) - htop      # the number of lines on the bottom
    
    print(charset[0] + charset[1] * (width - 2) + charset[3])

    for i in range(htop):
        print(charset[2] + " " * (width - 2) + charset[2] )

    for line in text:
        w2l = (width - 2 - len(line)) // 2      # the number of spaces on the left
        w2r = (width - 2 - len(line)) - w2l     # the number of spaces on the right
        print(charset[2] + (" " * w2l) + line + (" " * w2r) + charset[2])  # print the line

    for i in range(hbot):  # the remaining lines (if height is odd, the last line is added here)
        print(charset[2] + " " * (width - 2) + charset[2] )

    print(charset[4] + charset[1] * (width - 2) + charset[5])



# Instructions: 
# Run the following code to test the function being in the parent folder
#
#   py -m mylib.text
#
# the parameter -m is used to run a module as a script
#
# When using the -m parameter, the module name must be specified, not the file name. 
# You don't need to specify the .py extension nor the path.
# The module name is the name of the folder containing the __init__.py file, 
# followed by the name of the file without the .py extension, just like the import statement.
#
# You can also change to the mylib folder and run the script with the following command:
#
#   cd mylib
#   py text.py
#
if __name__ == "__main__":
    box("Hello World!", width=60, height=5)
    box("Hello World!", width=60, height=5, charset='=')
    box("Hello World!", width=60, height=5, charset='B')
    box("Hello World!", width=60, height=5, charset='b')
    box("Hello World!", width=60, height=5, charset='+')
    box("Hello World!", width=60, height=5, charset='#')
    box("Hello World!", width=60, height=5, charset='*')
