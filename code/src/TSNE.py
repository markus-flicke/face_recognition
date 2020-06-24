import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from src import Config
import os

from sklearn import manifold
import seaborn as sns
import pickle
import numpy as np

from src.Recommender.SVMRecommender import rank_papers


def _load_encoding(X_filename):
    with open(os.path.join(Config.ENCODING_DIRECTORY, X_filename), 'rb') as f:
        X = pickle.load(f)
    y = np.zeros(X.shape[0])
    y[:82] = 1
    return X, y

def _load_abstracts(filename = os.path.join(Config.ENCODING_DIRECTORY, 'cscv_abstracts.pkl')):
    with open(filename, 'rb') as f:
        abstracts = pickle.load(f)
    return abstracts


def apply_tsne(X):
    method = manifold.TSNE(n_components=2,
                           init='pca',
                           early_exaggeration=12)
    Y = method.fit_transform(X.toarray())
    return Y


def plot(X_filename, names=None):
    abstracts = _load_abstracts()
    X, y = _load_encoding(X_filename)
    ranks = rank_papers(X, y)
    if names is None:
        names = ranks

    # setting colors: geiger, cvgr, ranking
    palette = sns.color_palette("hls", 3)
    colors = np.array([palette[2]] * 82 + [palette[1]] * (X.shape[0] - 82))
    colors[ranks] = palette[0]

    # Carry out t-SNE projection only if not stored from last run
    Y_filename = 'TSNE' + X_filename[1:]
    if Y_filename in os.listdir(Config.ENCODING_DIRECTORY):
        with open(os.path.join(Config.ENCODING_DIRECTORY, Y_filename), 'rb') as f:
            Y = pickle.load(f)
    else:
        Y = apply_tsne(X)
        with open(os.path.join(Config.ENCODING_DIRECTORY, Y_filename), 'wb') as f:
            pickle.dump(Y, f)

    # PLOTTING
    c = colors
    x = Y[:, 0]
    y = Y[:, 1]

    # Scatterplot
    fig, ax = plt.subplots(figsize=(50, 30))
    plt.title(f't-SNE projection of {X_filename}', size=30)
    plt.margins(0.1)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    sc = plt.scatter(x, y, c=c, s=100, marker='x')
    plt.legend(labels = ['geiger group'])

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
        title = names[indices_at_location[0]]
        abstract = abstracts.get(title)

        annot.set_text(f'{title}\n\n{abstract}')

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


if __name__ == '__main__':
    geiger_paper_directory = Config.GEIGER_TEXT_DIRECTORY
    geiger_paper_paths = [os.path.join(geiger_paper_directory, filename) for filename in os.listdir(geiger_paper_directory)]
    graphics_paper_directory = Config.ARXIV_VISION_TEXT_DIRECTORY
    graphics_paper_paths = [os.path.join(graphics_paper_directory, filename) for filename in os.listdir(graphics_paper_directory)]
    paper_paths = geiger_paper_paths + graphics_paper_paths
    paper_names = [path.split('\\')[-1][:-4] for path in paper_paths]

    plot('X_geiger_cscv.pkl', names=paper_names)
