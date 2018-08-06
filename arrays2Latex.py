'''
Created on 
Print numpy arrays to latex tables
Taken from Peter Fischer 26.02.2016
Modified by Pierros Ntelis 24.04.2018
'''

from __future__ import print_function, division
from numpy import *

def print_to_table(avgs, stds, rows, cols,
                   highlight_best=True,
                   max_is_best=False,
                   avg_format='{0:!>4.1f}',
                   std_format='{1:!<4.1f}',
                   file=None,
                   col_orientation='c'):
    """
    Print data into a LateX table
    The data is given as averages and stadnard deviations.
    It will be printed as 'avgs' \pm 'stds'
    The names of the rows are given in 'rows' and the names of the columns and
    keys to the data dictionaries are given in 'cols'.
    avgs : dict of numpy array with average values
           each numpy array has shape  (N,)
    stds : dict of numpy arrays with standard deviations
           each numpy array shape (N,)
    rows: list of row names, length (N,)
    cols: list of column names and keys to dicts, length (N,)
    highlight_best: best values are printed in bold face, boolean
    max_is_best: True or False 
                 True:  largest values are best
                 False: the value with the smallest precision are best 
    avg_format: python string format specifier for printing average values,
                fill-value '!' will be replaced by \phantom{0} in LateX-code,
                since spaces are not respected in LateX code
    std_format: python string format specifier for printing standard deviations,
                see avg_format
    file: file to which the output should be written, string or None
          If file is None, table is printed to stdout
    """
    # Find best values
    best_avg = {}
    if highlight_best:
        if max_is_best:
            for c in cols:
                idx = np.argmax(avgs[c], axis=0)
                best_avg[c] = idx
        else:
            for c in cols:
                idx = np.argmin(array(stds[c])/array(avgs[c]), axis=0)
                best_avg[c] = idx

    # Header
    print_str = '\\begin{tabular}{l' + (" "+col_orientation) * len(cols) + '} \hline\n'
    for c in cols:
        print_str += ' & ' + c
    print_str += ' \\\\ \n \hline'

    # Content
    for m in range(len(rows)):
        row_name = rows[m]
        print_str += '\n{0}'.format(row_name)
        for c in cols:
            print_str += ' &\n '
            print_list = [avgs[c][m]]
            print_list += [stds[c][m]]
            if m == best_avg.get(c, None):
                print_str += ('$\mathbf{{'+ avg_format + ' \\pm ' + std_format + '}}$').format(*print_list)
            else:
                print_str += ('$'+ avg_format + ' \\pm ' + std_format + '$').format(*print_list)

        print_str += "\n \\vspace*{0.0mm} \\\\"
        print_str = print_str.replace('!', '\\phantom{0}')

    # Footer
    print_str += '\n \hline \n \end{tabular}'

    # Write output
    if file is None:
        print(print_str)
    else:
        with open(file, 'w') as f:
            f.write(print_str)

if __name__ == '__main__':
    '''
    Random test case
    '''
    import numpy as np
    # Load results from file
    cols = ['column 1', 'column 2']
    signal_avg = {}
    signal_std = {}
    N = 10
    for c in cols:
        signal_avg[c] = np.random.rand(N)
        signal_std[c] = np.random.rand(N) * 0.5

    # print to table
    names = ['row ' + str(m) for m in range(N)]
    print_to_table(signal_avg, signal_std, names, cols, avg_format='{0:!>0.3f}', std_format='{1:!>0.3f}')


print 'to produce the Latex Tables pntelis et al 2018 use one out of the two options below: OPTION1,OPTION2'

"""
#### z2, OPTION1
cols = ['bias','ocdm','Omega_k']
signal_avg = {}
signal_std = {}
for c in cols:
    signal_avg[c] = [chains_all[i]['chains'].item()[c].mean()  for i in range(len(chains_all))]
    signal_std[c] = [chains_all[i]['chains'].item()[c].std()  for i in range(len(chains_all))]
print_to_table(signal_avg, signal_std, names, cols, avg_format='{0:!>0.3f}', std_format='{1:!>0.3f}')

#### ALL z, OPTION2, more important test
cols = ['bias0','bias1','bias2','bias3','bias4','ocdm','Omega_k']
signal_avg = {}
signal_std = {}
for c in cols:
    signal_avg[c] = [chains_all[i]['chains'].item()[c].mean()  for i in range(len(chains_all))]
    signal_std[c] = [chains_all[i]['chains'].item()[c].std()  for i in range(len(chains_all))]
print_to_table(signal_avg, signal_std, names, cols, avg_format='{0:!>0.3f}', std_format='{1:!>0.3f}')
"""