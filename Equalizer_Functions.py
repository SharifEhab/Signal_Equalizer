import  streamlit_vertical_slider  as svs

def generate_slider(dict_values, values_slider):
    slider_values = []
    for i, (label, value_range) in enumerate(dict_values):
        slider_val = svs.vertical_slider(key=f"slider_{i}",min_value=value_range[0], max_value=value_range[1], step=values_slider[i][2])
        slider_values.append(slider_val)
    return slider_values