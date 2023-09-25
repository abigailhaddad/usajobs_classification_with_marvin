# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:11:29 2023

@author: abiga
"""

import pandas as pd
import matplotlib.pyplot as plt

# Aggregate data
agg_a = df['job_title'].str.title().value_counts().to_dict()
agg_b = df['positionTitle'].str.title().value_counts().to_dict()


def generate_final_comparison_plot_width_adjusted(data_a, data_b, n, figsize=(12, 6)):
    """
    Generate a final comparison plot for two processes with width adjustments.
    """
    # Sort the data and take the top n values
    sorted_data_a = dict(sorted(data_a.items(), key=lambda item: item[1], reverse=True)[:n])
    sorted_data_b = dict(sorted(data_b.items(), key=lambda item: item[1], reverse=True)[:n])

    def draw_words(ax, data, color, ha='center'):
        """
        Draw words with final refinements:
        1. Add more space between words.
        2. Return the total height taken by the words for dynamic image scaling.
        """
        y_pos = 1  # Starting position
        total_height = 0

        for word, count in data.items():
            font_size = 15 + (count / sum(data.values())) * 50  # Base size + scaling based on count
            ax.text(0.5 if ha == 'center' else 0, y_pos, word, ha=ha, va='top', color=color, fontsize=font_size)
            
            # Calculate the decrement in y_pos based on the font size and add a buffer for spacing
            decrement = (font_size / 350) + 0.01  # Increased spacing and adjusted for better scaling
            y_pos -= decrement
            total_height += decrement

        ax.set_xlim(0, 1)
        ax.axis('off')
        
        return total_height

    # Calculate the total heights for Processes A and B for dynamic image scaling
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    height_a = draw_words(axs[0], sorted_data_a, 'blue', ha='left')
    height_b = draw_words(axs[1], sorted_data_b, 'red', ha='left')
    max_height = max(height_a, height_b)

    # Adjust figure dimensions based on content
    fig.set_size_inches(figsize[0]*1.5, max_height * 10)  # Increased width scaling for better visualization

    axs[0].set_title(f"LLM-Generated Titles", fontsize=20)
    axs[1].set_title(f"Official Titles", fontsize=20)
    plt.tight_layout()
    plt.show()


# Test the final function with n=3
generate_final_comparison_plot_width_adjusted(agg_a, agg_b, n=3)

