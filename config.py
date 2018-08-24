"""
In this module we can force other defaults from the default values.
Simply change the corresponding value to your preference and run apple.py
"""


class ApplePickerConfig(object):
    particle_size = None
    query_image_size = None
    query_window_size = None
    max_particle_size = None
    min_particle_size = None
    minimum_overlap_amount = None
    tau1 = None
    tau2 = None
    container_size = 450
    proc = 1
    output_dir = None
