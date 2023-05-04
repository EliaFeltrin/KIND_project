import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib import rc

SPIDER_ROW_N = 2
SPIDER_COL_N = 2

def to_lowerCase(df):
    return pd.DataFrame({'Token': df['Token'].str.lower(), 'Entity': df['Entity']})

def add_column_names(df):
    return  df.rename(columns={0: 'Token', 1: 'Entity'})

def spider_plot(df, group, title, subplot_idx):
    categories=list(df)[:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    print(angles)
    
    # Initialise the spider plot
    ax = plt.subplot(SPIDER_ROW_N, SPIDER_COL_N, subplot_idx, polar=True)
    
    # first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    #ax.set_yscale('log')
    ax.set_rlabel_position(0)
    min = df.min().min()
    max = df.max().max()
    plt.ylim(min -(max-min)/10, max + (max-min)/10)
 
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
    
    for i in range(len(group)):
        values=df.loc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=group[i])
        ax.fill(angles, values, 'b', alpha=0.1)

    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title)

    # Show the graph
    #plt.show()

############################################################### reading datasets ###############################################################

ds = {'ds_mr' : pd.read_csv('./KIND_project/dataset/KIND-main/dataset/moro_train.tsv', sep='\t', header=None),
      'ds_mr_test' : pd.read_csv('./KIND_project/dataset/KIND-main/dataset/moro_test.tsv', sep='\t', header=None),
      
      'ds_dg_IOB' : pd.read_csv('./KIND_project/dataset/KIND-main/evalita-2023/ADG_train.tsv', sep='\t', header=None),
      'ds_fc_IOB' : pd.read_csv('./KIND_project/dataset/KIND-main/evalita-2023/FIC_train.tsv', sep='\t', header=None),
      'ds_wn_IOB' : pd.read_csv('./KIND_project/dataset/KIND-main/evalita-2023/WN_train.tsv', sep='\t', header=None),
      
      'ds_dg_IOB_test' : pd.read_csv('./KIND_project/dataset/KIND-main/evalita-2023/ADG_dev.tsv', sep='\t', header=None),
      'ds_fc_IOB_test' : pd.read_csv('./KIND_project/dataset/KIND-main/evalita-2023/FIC_dev.tsv', sep='\t', header=None),
      'ds_wn_IOB_test' : pd.read_csv('./KIND_project/dataset/KIND-main/evalita-2023/WN_dev.tsv', sep='\t', header=None)
}

############################################################### calculating stats ###############################################################
stats = {}

for i in ds.keys():     
    ds[i] = add_column_names(ds[i])
    ds[i] = to_lowerCase(ds[i])

    stats[i] = {
        'doc_len' : ds[i]['Token'].count(),
        'voc_size' : ds[i]['Token'].nunique(),
        'n_punct': sum(1 for k in ds[i]['Token'] if all(char in string.punctuation for char in k))
    }

    if('IOB' in str(i)):
        
        stats[i]['n_I-PER'] = sum(1 for k in ds[i]['Entity'] if k == 'I-PER') 
        stats[i]['n_I-ORG'] = sum(1 for k in ds[i]['Entity'] if k == 'I-ORG')
        stats[i]['n_I-LOC'] = sum(1 for k in ds[i]['Entity'] if k == 'I-LOC')

        stats[i]['n_B-PER'] = sum(1 for k in ds[i]['Entity'] if k == 'B-PER') 
        stats[i]['n_B-ORG'] = sum(1 for k in ds[i]['Entity'] if k == 'B-ORG')
        stats[i]['n_B-LOC'] = sum(1 for k in ds[i]['Entity'] if k == 'B-LOC')

        stats[i]['n_PER'] = stats[i]['n_I-PER'] + stats[i]['n_B-PER']
        stats[i]['n_ORG'] = stats[i]['n_I-ORG'] + stats[i]['n_B-ORG']
        stats[i]['n_LOC'] = stats[i]['n_I-LOC'] + stats[i]['n_B-LOC']
        
    else:
        stats[i]['n_PER'] = sum(1 for k in ds[i]['Entity'] if k == 'PER') 
        stats[i]['n_ORG'] = sum(1 for k in ds[i]['Entity'] if k == 'ORG')
        stats[i]['n_LOC'] = sum(1 for k in ds[i]['Entity'] if k == 'LOC')

    stats[i]['n_O'] = sum(1 for k in ds[i]['Entity'] if k == 'O')
        

avg_doc_len = sum(stats[i]['doc_len'] for i in stats.keys()) / len(stats.keys())
avg_voc_size = sum(stats[i]['voc_size'] for i in stats.keys()) / len(stats.keys())

############################################################### preparing data for plotting ###############################################################
 
# Values of each group
iper = [stats[i]['n_I-PER'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]
iorg = [stats[i]['n_I-ORG'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]
iloc = [stats[i]['n_I-LOC'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]

bper = [stats[i]['n_B-PER'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]
borg = [stats[i]['n_B-ORG'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]
bloc = [stats[i]['n_B-LOC'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]

iob_punct = [stats[i]['n_punct'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]
iob_o = [stats[i]['n_O'] for i in stats.keys() if 'IOB' in str(i) and 'test' not in str(i)]
iob_o = [iob_o[i] - iob_punct[i] for i in range(len(iob_o))]

iper_test = [stats[i]['n_I-PER'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]
iorg_test = [stats[i]['n_I-ORG'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]
iloc_test = [stats[i]['n_I-LOC'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]

bper_test = [stats[i]['n_B-PER'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]
borg_test = [stats[i]['n_B-ORG'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]
bloc_test = [stats[i]['n_B-LOC'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]

iob_punct_test = [stats[i]['n_punct'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]
iob_o_test = [stats[i]['n_O'] for i in stats.keys() if 'IOB' in str(i) and 'test' in str(i)]
iob_o_test = [iob_o_test[i] - iob_punct_test[i] for i in range(len(iob_o_test))]

iob_voc_size = [stats[i]['voc_size'] for i in stats.keys() if 'IOB' in str(i)]

per = [stats[i]['n_PER'] for i in stats.keys() if 'test' not in str(i)]
org = [stats[i]['n_ORG'] for i in stats.keys() if 'test' not in str(i)]
loc = [stats[i]['n_LOC'] for i in stats.keys() if 'test' not in str(i)]

per_test = [stats[i]['n_PER'] for i in stats.keys() if 'test' in str(i)]
org_test = [stats[i]['n_ORG'] for i in stats.keys() if 'test' in str(i)]
loc_test = [stats[i]['n_LOC'] for i in stats.keys() if 'test' in str(i)]

voc_size = [stats[i]['voc_size'] for i in stats.keys() if 'test' not in str(i)]
voc_size_test = [stats[i]['voc_size'] for i in stats.keys() if 'test' in str(i)]

doc_len = [stats[i]['doc_len'] for i in stats.keys() if 'test' not in str(i)]
doc_len_test = [stats[i]['doc_len'] for i in stats.keys() if 'test' in str(i)]

n_punct = [stats[i]['n_punct'] for i in stats.keys() if 'test' not in str(i)]
n_punct_test = [stats[i]['n_punct'] for i in stats.keys() if 'test' in str(i)]

o = [stats[i]['n_O'] for i in stats.keys() if 'IOB' not in str(i)]
punct = [stats[i]['n_punct'] for i in stats.keys() if 'IOB' not in str(i)]


############################################################### spider plots ###############################################################

spider_plot( pd.DataFrame({
    'B-PER': bper,
    'I-PER': iper,
    'B-ORG': borg,
    'I-ORG': iorg,
    'B-LOC': bloc,
    'I-LOC': iloc }),
    ['deGasperi', 'Fiction', 'Wikinews'],
    'Train set IOB tags',
    1)

spider_plot( pd.DataFrame({
    'B-PER': bper_test,
    'I-PER': iper_test,
    'B-ORG': borg_test,
    'I-ORG': iorg_test,
    'B-LOC': bloc_test,
    'I-LOC': iloc_test }),
    ['deGasperi', 'Fiction', 'Wikinews'],
    'Test set IOB tags',
    2)

spider_plot( pd.DataFrame({
    'B-PER': per,
    'B-ORG': org,
    'B-LOC': loc}),
    ['Moro', 'deGasperi', 'Fiction', 'Wikinews'],
    'Train set non-IOB tags',
    3)

spider_plot( pd.DataFrame({
    'B-PER': per_test,
    'B-ORG': org_test,
    'B-LOC': loc_test}),
    ['Moro', 'deGasperi', 'Fiction', 'Wikinews'],
    'Test set non-IOB tags',
    4)

#spider_plot( pd.DataFrame({
#    'doc_len': doc_len,
#    'voc_size': voc_size,
#    'n_punct': n_punct,}),
#    ['Moro', 'deGasperi', 'Fiction', 'Wikinews'],
#    'Train set document statistics',
#    5)
#
#spider_plot( pd.DataFrame({
#    'doc_len': doc_len_test,
#    'voc_size': voc_size_test,
#    'n_punct': n_punct_test,}),
#    ['Moro', 'deGasperi', 'Fiction', 'Wikinews'],
#    'Test set document statistics',
#    6)


plt.show()


def stacked_bar_plot(names, data):
    
    # Names of group and bar width
    barWidth = 1
    bars = []
    n_col = np.arange(len(names))

    for i in range(len(names)):
        plt.bar(n_col, data[i], bottom=bars, edgecolor='white', width=barWidth, label=names[i])
        bars = np.add(bars, data[i]).tolist()

    bars = np.add(bper, iper).tolist()

    
    # Custom X axis
    plt.xticks(r, names, fontweight='bold')
    plt.xlabel("group")
    plt.legend()
    #ax.set_yscale('log')
    #ax.set_ybound(1, 1000000)
    
    # Show graphic
stacked_bar_plot(['doc_len', 'voc_size', 'n_punct', 'n_O'],
                 [doc_len, voc_size, n_punct, o])
plt.show()