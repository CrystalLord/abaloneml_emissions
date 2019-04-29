"""
Gets the name of the columns.
"""

def get_col_name(name_file, col_num):
    with open(name_file, 'r') as f:
        names = f.readline().split(",")
    return names[col_num]


def get_index(X, name_file, column_names):
    name_to_ind = {}
    with open(name_file, 'r') as f:
        names = f.readline().split()
        for i, n in enumerate(names):
            name_to_ind[n] = i
    index_list = [name_to_ind[n] for n in  column_names]
    return X[index_list,:]


