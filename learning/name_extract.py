"""
Gets the name of the columns.
"""

def get_col_name(name_file, colnum):
    with open(name_file, 'r') as f:
        names = f.readline().split()
    return names[colnum]
