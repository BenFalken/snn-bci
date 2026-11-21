"""
The paper's figure style, shared by every figure of the three experiments (motor, audio, speech).

Sans-serif journal style: Helvetica/Arial, white, thin marks, recessive axes; widths in journal columns. Every
decoder has one fixed color in every figure. The core decoders (in every experiment) use an Okabe-Ito set that passes
the dataviz palette checks in this order (CVD separation >= 13 between neighbours; sky blue and orange are under 3:1
contrast on white, so every mark is labelled by name); the LSNN and the Speck SNN are the two highlights.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 89 * MM            # one journal column, inches
DOUBLE = 183 * MM           # two columns

CORE = ["kf", "wf", "lstm", "lsnn", "speck"]
EXTRA = ["gru", "qrnn", "paper_rnn", "delaynet", "dense_nn", "simple_rnn", "wiener_cascade"]
LABEL = {"kf": "Kalman filter", "wf": "Wiener filter", "lstm": "LSTM", "lsnn": "LSNN", "speck": "Speck SNN",
         "gru": "GRU", "aegru": "Autoencoder GRU", "qrnn": "QRNN", "paper_rnn": "Paper RNN", "delaynet": "DelayNet", "dense_nn": "Dense NN",
         "simple_rnn": "Simple RNN", "wiener_cascade": "Wiener cascade"}
SHORT = {"kf": "KF", "wf": "WF", "lstm": "LSTM", "lsnn": "LSNN", "speck": "Speck",
         "gru": "GRU", "aegru": "AE-\nGRU", "qrnn": "QRNN", "paper_rnn": "Paper\nRNN", "delaynet": "DelayNet", "dense_nn": "DNN",
         "simple_rnn": "sRNN", "wiener_cascade": "WC"}
COLOR = {"kf": "#56B4E9", "wf": "#009E73", "lstm": "#0072B2", "lsnn": "#D55E00", "speck": "#E69F00",
         # supplementary decoders: muted, so the core keeps the attention
         "gru": "#7A9CC6", "aegru": "#5f86b8", "qrnn": "#9C89B8", "paper_rnn": "#4D4D4D", "delaynet": "#C49A6C",
         "dense_nn": "#A0A0A0", "simple_rnn": "#8DB3A3", "wiener_cascade": "#B8B8B8"}
HIGHLIGHT = {"lsnn", "speck"}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

EXPERIMENTS = {"motor": "Motor: hand velocity", "jenkins": "Motor, monkey J: hand velocity, 192 channels", "audio": "Audio: heard speech", "speech": "Speech: attempted speech"}


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.2, "savefig.dpi": 300, "figure.dpi": 150,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def panel_label(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, path):
    """PDF, SVG and a PNG preview."""
    for ext in (".pdf", ".svg", ".png"):
        fig.savefig(str(path) + ext, bbox_inches="tight", facecolor="white")
    plt.close(fig)
