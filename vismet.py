# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:16:26 2025
VisMet: Visualization Metrology
library with tools for visualization of the Campaign Network.
@author: mar
"""

import os
import yaml
from fpdf import FPDF

def export_paths_to_pdf(paths, output_file="oscillator_paths.pdf", 
                        country_config_file="country_color.yml"):
    """
    function to generate a pdf with all the path between all the comparisons
    between oscillators.

    Parameters
    ----------
    paths (dict): contains the information of the path between oscillators
    
    output_file (str): Name of the output file. The default is "oscillator_paths.pdf".
        
    country_config_file : str. File with color information of oscillators
    by country. The default is "country_color.yml".

    Returns
    -------
    None.

    """
    color_lookup = {}
    if country_config_file and os.path.isfile(country_config_file):
        with open(country_config_file, 'r') as f:
            country_config = yaml.safe_load(f)
        for country, info in country_config.get('countries', {}).items():
            color = info.get('color', '#000000')
            for inst in info.get('institutes', []):
                color_lookup[inst] = color

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="TOCK 2025 Network Paths", ln=True, align='C')
    pdf.ln(10)

    max_per_line = 16
    x_spacing = 16
    ellipse_diameter = 6

    for (osc1, osc2), info in paths.items():
        path = info['path']

        # Subtitle
        pdf.set_font("Arial", style='B', size=9)
        pdf.cell(0, 10, txt=f"{osc1} -> {osc2}", ln=True)
        pdf.set_font("Arial", size=6)

        # Initialize drawing coords
        y_base = pdf.get_y() + 5
        x_start = 10
        x_pos = x_start
        items_in_line = 0

        for idx, node in enumerate(path):
            inst = node.split('_')[0]
            color = color_lookup.get(inst, '#000000')
            r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            pdf.set_draw_color(r, g, b)
            pdf.set_fill_color(r, g, b)
            pdf.ellipse(x_pos, y_base, ellipse_diameter, ellipse_diameter, style='FD')
            pdf.set_text_color(0, 0, 0)

            # Trim node name to INST_OSC
            parts = node.split('_')
            trimmed_node = '_'.join(parts[:2]) if len(parts) >= 2 else node
            inst_part, osc_part = trimmed_node.split('_', 1)
            label = f"{inst_part} \n {osc_part}"

            # Draw multiline text manually
            pdf.set_xy(x_pos - 5, y_base - 6)
            pdf.multi_cell(ellipse_diameter + 10, 3, txt=label, align='C')

            # Draw arrow line
            if idx < len(path) - 1:
                x_line_start = x_pos + ellipse_diameter
                x_line_end = x_pos + x_spacing
                pdf.line(x_line_start, y_base + 3, x_line_end, y_base + 3)

            x_pos += x_spacing
            items_in_line += 1

            if items_in_line >= max_per_line:
                y_base += 15
                x_pos = x_start
                items_in_line = 0

        pdf.set_y(y_base + 10)

    pdf.output(output_file)
    print(f"✅ Paths exported to {output_file}")
