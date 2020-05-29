import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import seaborn as sns
import numpy as np


def plot_TSNE(X, y, colors = None):
    if not colors:
        colors_palette = sns.color_palette('hls', np.unique(y).shape[0])
        color_idx = LabelEncoder().fit_transform(y)
        colors = [colors_palette[i] for i in color_idx]

    tsne = TSNE(n_components=2,
                init='pca',
                early_exaggeration=12)
    X_2D = tsne.fit_transform(X)

    x_plot = X_2D[:, 0]
    y_plot = X_2D[:, 1]

    # Scatterplot
    fig, ax = plt.subplots(figsize=(50, 30))
    plt.title(f'T-SNE', size=30)
    plt.margins(0.1)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    sc = plt.scatter(x_plot, y_plot, c=colors, s=200)

    # Annotations
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    annot.set_fontsize(12)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        indices_at_location = ind["ind"]
        # Making the Annotation Text
        label = y[indices_at_location[0]]
        annot.set_text(f'{label}')

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    # maximising only works in jupyter for now
    # somehow connected to using the 'qt backend'
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        pass
    plt.show()
