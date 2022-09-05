from logging import Logger
from random import randint

import numpy as np
from matplotlib import patches, pyplot


def assignmentArray_to_lists(assignment_array):
    from collections import defaultdict

    by_attribute_value = defaultdict(list)
    for node_index, attribute_value in enumerate(assignment_array):
        by_attribute_value[attribute_value].append(node_index)
    return by_attribute_value.values()


def draw_adjacency_matrix(A, partitions=[], output_file="", logger=Logger):
    """
    - A is the adjacency matrix
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """

    colors = []

    for i in range(len(partitions)):
        colors.append("%06X" % randint(0, 0xFFFFFF))
    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(10, 10))
    pyplot.imshow(A, cmap="Greys", vmin=0, vmax=5)
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition in [partitions]:
        current_idx = 0
        for module, color in zip(partition, colors):
            ax.add_patch(
                patches.Rectangle(
                    (current_idx, current_idx),
                    len(module),  # Width
                    len(module),  # Height
                    alpha=1,
                    facecolor="none",
                    edgecolor="#" + color,
                    linewidth="1.5",
                )
            )
            current_idx += len(module)
    fig.savefig(output_file)
    logger.info(f"Figure saved at: output_file")
