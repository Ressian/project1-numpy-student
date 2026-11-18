"""Generate the figures used in the experiment report (matplotlib, no Chinese font needed)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = 'figs/'

# ---------- Figure 1: patchify reshape/transpose ----------
def fig_patchify():
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')

    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                                    fc=fc, ec='#333', lw=1))
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=10,
                family='monospace')

    box(0.2, 1.3, 2.4, 1.4, '[B,3,224,224]\ninput image', '#dbeafe')
    ax.add_patch(FancyArrowPatch((2.7, 2.0), (3.5, 2.0), arrowstyle='->', lw=1.6,
                                 mutation_scale=15))
    ax.text(3.1, 2.35, 'reshape', ha='center', fontsize=9)
    ax.text(3.1, 1.65, '(7,32)', ha='center', fontsize=8, color='#555')

    box(3.6, 1.0, 2.7, 2.0, '[B,3,7,32,7,32]\nB,C,gh,ph,gw,pw', '#fef3c7')
    ax.add_patch(FancyArrowPatch((6.4, 2.0), (7.2, 2.0), arrowstyle='->', lw=1.6,
                                 mutation_scale=15))
    ax.text(6.8, 2.35, 'transpose', ha='center', fontsize=9)
    ax.text(6.8, 1.65, '(0,2,4,1,3,5)', ha='center', fontsize=8, color='#555')

    box(7.3, 1.0, 2.5, 2.0, '[B,7,7,3,32,32]\nB,gh,gw,C,ph,pw', '#dcfce7')
    ax.text(5.0, 3.5, 'TODO-01  patchify: cut 7x7 patches, flatten each 32x32x3 into 3072',
            ha='center', fontsize=10, weight='bold', color='#b91c1c')
    ax.text(5.0, 0.3, 'then reshape -> [B,49,3072]  (row-major: gh first, gw second)',
            ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG + 'patchify.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------- Figure 2: full pipeline ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

    stages = [
        (0.3, 'Image\n[B,3,224,224]', '#dbeafe'),
        (2.1, 'Patches\n[B,49,3072]', '#fef3c7'),
        (4.0, 'Tokens\n[B,50,768]', '#dcfce7'),
        (5.9, '12 Blocks\n[B,50,768]', '#f3e8ff'),
        (7.8, 'CLS\n[B,768]', '#fef3c7'),
        (9.0, 'Output\n[B,512]', '#dbeafe'),
    ]
    widths = [1.4, 1.6, 1.6, 1.6, 1.0, 0.9]
    centers = []
    for (x, txt, fc), w in zip(stages, widths):
        ax.add_patch(FancyBboxPatch((x, 2.0), w, 1.2, boxstyle='round,pad=0.02',
                                    fc=fc, ec='#333', lw=1))
        ax.text(x + w/2, 2.6, txt, ha='center', va='center', fontsize=8.5,
                family='monospace')
        centers.append((x, w))
    for i in range(len(centers)-1):
        xe = centers[i][0] + centers[i][1]
        xs = centers[i+1][0]
        ax.add_patch(FancyArrowPatch((xe+0.02, 2.6), (xs-0.02, 2.6),
                                     arrowstyle='->', lw=1.5, mutation_scale=13))

    labels = ['patchify', 'embed\n+CLS', 'pre_norm\n+blocks', 'take\nCLS', 'proj\n+unit']
    for i, lab in enumerate(labels):
        xe = centers[i][0] + centers[i][1]
        xs = centers[i+1][0]
        ax.text((xe+xs)/2, 3.45, lab, ha='center', fontsize=8, color='#1d4ed8')

    ax.text(5.0, 4.5, 'OpenCLIP ViT-B-32 image encoder (NumPy forward)',
            ha='center', fontsize=12, weight='bold')
    ax.text(1.1, 0.9, 'TODO-01', ha='center', fontsize=9, color='#b91c1c', weight='bold')
    ax.text(2.9, 0.9, 'TODO-05', ha='center', fontsize=9, color='#b91c1c', weight='bold')
    ax.text(4.8, 0.9, 'TODO-02/03/04 in blocks', ha='center', fontsize=8, color='#b91c1c')
    ax.text(7.2, 0.9, 'TODO-06', ha='center', fontsize=9, color='#b91c1c', weight='bold')
    fig.tight_layout()
    fig.savefig(FIG + 'pipeline.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------- Figure 3: one transformer block ----------
def fig_block():
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_xlim(0, 6); ax.set_ylim(0, 10); ax.axis('off')

    def box(x, y, w, h, txt, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                                    fc=fc, ec='#333', lw=1))
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs,
                family='monospace')

    box(2.1, 9.0, 1.8, 0.7, 'x [B,50,768]', '#dbeafe')
    box(2.1, 7.4, 1.8, 0.7, 'norm1', '#fef3c7')
    box(2.1, 5.8, 1.8, 0.7, 'attention\n(12 heads)', '#dcfce7')
    box(2.1, 4.2, 1.8, 0.7, '+ x  (residual 1)', '#fee2e2')
    box(2.1, 2.9, 1.8, 0.7, 'norm2 -> u', '#fef3c7')
    box(2.1, 1.5, 1.8, 0.7, 'MLP\n768->3072->768', '#f3e8ff')
    box(2.1, 0.2, 1.8, 0.7, '+ residual (res 2)', '#fee2e2')

    ys = [9.0, 7.4, 5.8, 4.2, 2.9, 1.5, 0.2]
    for i in range(len(ys)-1):
        y0 = ys[i]; y1 = ys[i+1]+0.7
        ax.add_patch(FancyArrowPatch((3.0, y0), (3.0, y1), arrowstyle='->',
                                     lw=1.5, mutation_scale=13))
    # residual skips
    ax.plot([1.9, 0.7, 0.7, 1.9], [9.35, 9.35, 4.55, 4.55], color='#dc2626', lw=1.4)
    ax.text(0.15, 7.0, 'skip', fontsize=8, color='#dc2626', rotation=90, va='center')
    ax.plot([1.9, 0.7, 0.7, 1.9], [4.55, 4.55, 0.55, 0.55], color='#dc2626', lw=1.4)
    ax.text(0.15, 2.5, 'skip', fontsize=8, color='#dc2626', rotation=90, va='center')

    ax.text(4.3, 6.1, 'TODO-03', fontsize=9, color='#b91c1c', weight='bold')
    ax.text(4.3, 1.8, 'TODO-04b', fontsize=9, color='#b91c1c', weight='bold')
    ax.text(4.3, 4.5, 'TODO-04a', fontsize=9, color='#b91c1c', weight='bold')
    ax.set_title('One TransformerBlock (repeated 12 times)', fontsize=11, weight='bold')
    fig.tight_layout()
    fig.savefig(FIG + 'block.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_patchify()
    fig_pipeline()
    fig_block()
    print('figures written to', FIG)
