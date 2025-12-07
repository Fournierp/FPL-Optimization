import streamlit as st

from src.pages import (
    all_in_one,
    automated_chips,
    biased,
    differential,
    home,
    projection_data,
    select_chips,
    sensitivity_analysis,
    vanilla,
    wildcard,
)

st.set_page_config(page_title='FPL Optimization', page_icon='chart_with_upwards_trend')

PAGES = {
    'Home': home,
    'Projection Data': projection_data,
    'Vanilla': vanilla,
    'Differential': differential,
    'Biased': biased,
    'Wildcard': wildcard,
    'Select Chips': select_chips,
    'Automated Chips': automated_chips,
    'Sensitivity Analysis': sensitivity_analysis,
    'All In One': all_in_one,
}


def main() -> None:
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Visit', list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f'Loading {selection} ...'):
        page.write()


if __name__ == '__main__':
    main()
