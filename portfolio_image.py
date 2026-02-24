"""
PRAGYAM Portfolio Image Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a highly minimal, clean Matplotlib-based tabular image 
inspired by utility-first aesthetic design.
"""

import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ensures headless environment compatibility
import matplotlib.pyplot as plt
from datetime import datetime

def generate_portfolio_image(portfolio_df: pd.DataFrame, metadata: dict, max_rows: int = 30) -> bytes:
    """
    Generates a minimal table-style image of the portfolio using matplotlib.
    Returns PNG bytes.
    """
    if portfolio_df.empty:
        return _generate_error_image("No positions to display")

    df = portfolio_df.head(max_rows).copy()
    
    # ─── Configuration & Exact Height Sizing ───
    plt.style.use('dark_background')
    
    # Mathematically lock the heights to eliminate uncontrolled padding
    header_h = 0.55  # Inches for the top title area
    footer_h = 0.30  # Inches for the bottom footer area
    row_h = 0.35     # Inches per table row
    
    table_h = (len(df) + 1) * row_h  # +1 accounts for the header row
    fig_height = header_h + table_h + footer_h
    
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    
    bg_color = '#121212'
    fig.patch.set_facecolor(bg_color)
    
    # ─── Eliminate Padding (Subplots Adjust) ───
    # This locks the drawing area exactly between the header and footer bounds
    fig.subplots_adjust(
        top=1 - (header_h / fig_height),
        bottom=(footer_h / fig_height),
        left=0.02,
        right=0.98
    )
    
    # ─── Data Preparation ───
    table_data = []
    for i, (_, row) in enumerate(df.iterrows()):
        symbol = str(row.get('symbol', 'N/A'))
        price = f"₹{row.get('price', 0):,.2f}"
        units = f"{int(row.get('units', 0)):,}"
        weight = f"{row.get('weightage_pct', 0):.2f}%"
        value = f"₹{row.get('value', 0):,.0f}"
        
        table_data.append([i + 1, symbol, price, units, weight, value])
        
    headers = ["#", "Symbol", "Price", "Units", "Weightage", "Total Value"]
    
    # ─── Table Creation ───
    # bbox=[0, 0, 1, 1] forces the table to stretch entirely to the limits of our adjusted bounds,
    # destroying any remaining space above or below the table.
    table = ax.table(
        cellText=table_data, 
        colLabels=headers, 
        loc='center', 
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # ─── Colorizing & Cell Formatting ───
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#2A2A2A') # Soft border
        
        row_idx, col_idx = key
        
        if row_idx == 0:
            # Header Row
            cell.set_facecolor('#202020')
            cell.set_text_props(color='#FFC300', weight='bold') # Brand accent color
        else:
            # Data Rows (Alternating colors for readability)
            is_even_row = row_idx % 2 == 0
            cell.set_facecolor('#1E1E1E' if is_even_row else '#181818')
            cell.set_text_props(color='#EAEAEA')
            
            # Highlight high weightages slightly
            if col_idx == 4: # Weightage column
                val_str = table_data[row_idx-1][4].replace('%', '')
                try:
                    if float(val_str) > 4.5:
                        cell.set_text_props(color='#10B981') # Green for top picks
                except:
                    pass

    # ─── Footer Details ───
    style = metadata.get('investment_style', 'N/A')
    regime = metadata.get('regime', {}).get('name', 'N/A')
    now = datetime.now().strftime('%d %b %Y, %H:%M')
    
    footer_text = f"PRAGYAM | {style} | Regime: {regime} | {now}"
    
    # Placed dynamically in the exact middle of the footer height
    plt.figtext(
        0.5, (footer_h * 0.4 / fig_height), 
        footer_text, 
        ha='center', va='center', 
        color='#777777', 
        fontsize=9
    )
    
    # ─── Title Details ───
    capital = f"₹{metadata.get('capital', 0):,.0f}"
    
    # Placed dynamically in the exact middle of the header height
    plt.figtext(
        0.5, 1 - (header_h * 0.5 / fig_height), 
        f"Curated Portfolio — Capital: {capital}", 
        ha='center', va='center',
        color='#FFFFFF', 
        weight='bold',
        fontsize=14
    )
    
    # ─── Export ───
    buf = io.BytesIO()
    # Explicitly omitting bbox_inches='tight' forces matplotlib to respect our strict math layout
    plt.savefig(buf, format='PNG', facecolor=bg_color, dpi=300)
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()


def _generate_error_image(msg: str) -> bytes:
    """Fallback utility for empty states or errors."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    fig.patch.set_facecolor('#121212')
    
    plt.figtext(0.5, 0.5, f"⚠️ Error: {msg}", ha='center', va='center', color='#EF4444', fontsize=12)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', facecolor='#121212', dpi=300)
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()
