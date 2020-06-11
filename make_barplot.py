import numpy as np
import matplotlib.pyplot as plt

def make_barplots(ylabel, ymin, ymax, save_file):
    n_groups = 10
#     data to plot
    f_er = np.ones(10) * 0.2
    m_er = np.ones(10) * 0.3
    er = np.ones(10) * 0.25

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22

    rects = plt.bar(index, er, bar_width, label='Combined')
    rects_f = plt.bar(index + bar_width, f_er, bar_width, label='Female')
    rects_m = plt.bar(index + 2*bar_width, f_er, bar_width, label='Male')

    plt.ylabel(ylabel)
    plt.ylim(0, ylimit)
    plt.yticks(np.arange(ymin, ymax, 0.05))

    plt.xlabel('Experiment')
    xlabels = ['50/50', '20/80', '50/50 utt', '20/80 utt', '50/50 spk',
               '20/80 spk', '50/50 gndr', '20/80 gndr', '50/50 discrim',
               '20/80 discrim']
    plt.xticks(index + bar_width, xlabels, rotation=45)

    plt.title(f'{ylabel} by experiment and gender')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_file)
    plt.clf()

if __name__=='__main__':
    make_barplots('CER', 0, 0.5, 'cer.png')
