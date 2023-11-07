"""Constants used in visualisation module, e.g. colors, line widths etc."""

    

try:
    from manim import config, WHITE, GRAY_B, GRAY_C
    # manim configuration:
    config.background_color = WHITE
except:
    WHITE = '#FFFFFF'
    GRAY_B = 	'#777777'
    GRAY_C = '#444444'
     
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_hex



# graph constants:
radius_size = 0.36
arrow_tip_size = 0.5
buff_val = 0.30
stroke_width_thin = 4
stroke_width_medium = 5
stroke_width_thick = 6
max_stroke_width_to_length_ratio = 50

# legend constants:
legend_bar_height = 6
legend_bar_width = 0.3

# colors:
opacity_of_unused = 1
lightest_gray = '#F5F5F5'
medium_gray = GRAY_B
darkest_gray = GRAY_C
cmap = mpl.cm.viridis
colors = [to_hex(cmap(1.0)), to_hex(cmap(0.75)), to_hex(cmap(0.5)), to_hex(cmap(0.5)), to_hex(cmap(0.25)),
          to_hex(cmap(0.25)), to_hex(cmap(0.25)), to_hex(cmap(0.)), to_hex(cmap(0.)), to_hex(cmap(0.))]
color_map = LinearSegmentedColormap.from_list("color_map_viridis", colors)
