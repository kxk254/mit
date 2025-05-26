import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_model_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))

    # HGNetV2-based model diagram (top half)
    ax.text(1, 5.5, "HGNetV2-based Model", fontsize=12, fontweight='bold')

    ax.add_patch(patches.Rectangle((0.5, 5), 1, 0.5, facecolor='lightblue'))
    ax.text(1, 5.25, "HGNetV2\nBackbone", ha='center', va='center')

    ax.add_patch(patches.Rectangle((2, 5), 1, 0.5, facecolor='lightgreen'))
    ax.text(2.5, 5.25, "U-Net\nDecoder", ha='center', va='center')

    ax.add_patch(patches.Rectangle((3.5, 5), 1, 0.5, facecolor='salmon'))
    ax.text(4, 5.25, "Upsample\nHead", ha='center', va='center')

    ax.arrow(1.5, 5.25, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(3, 5.25, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # ConvNeXt-based model diagram (bottom half)
    ax.text(1, 3.5, "ConvNeXt-based Model", fontsize=12, fontweight='bold')

    ax.add_patch(patches.Rectangle((0.5, 3), 1, 0.5, facecolor='lightblue'))
    ax.text(1, 3.25, "ConvNeXt\n(modified)", ha='center', va='center')

    ax.add_patch(patches.Rectangle((2, 3), 1, 0.5, facecolor='lightgreen'))
    ax.text(2.5, 3.25, "Custom\nDecoder", ha='center', va='center')

    ax.add_patch(patches.Rectangle((3.5, 3), 1, 0.5, facecolor='orange'))
    ax.text(4, 3.25, "Head\n(no upsampling)", ha='center', va='center')

    ax.arrow(1.5, 3.25, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(3, 3.25, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(0, 6)
    ax.set_ylim(2.5, 6)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

draw_model_diagram()