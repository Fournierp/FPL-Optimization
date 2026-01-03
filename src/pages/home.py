import streamlit as st


def write() -> None:
    st.title('FPL - Home')
    with st.spinner('Loading About ...'):
        st.header('FPL Optimization.')
        st.write('Solve Optimal team for FPL.')

        st.info(
            """
            **Welcome to FPL Optimization!**

            This app helps you optimize your Fantasy Premier League team using mathematical optimization.

            **Available Pages:**
            - **Vanilla**: Basic optimization with standard constraints
            - **Differential**: Target low-ownership players for differential picks
            - **Biased**: Force specific players into your squad
            - **Wildcard**: Plan multi-gameweek strategies
            - **Select Chips**: Manually choose which chips to use and when
            - **Automated Chips**: Let the optimizer decide chip timing based on value
            - **Sensitivity Analysis**: Test robustness with Monte Carlo simulations
            - **All In One**: Combine chips, differential, and bias features

            **Getting Started:**
            1. Ensure your projection data is in `data/projections/` (CSV format)
            2. Choose a page from the sidebar
            3. Adjust parameters and constraints
            4. Run the optimization
            5. Review the suggested team and transfers
            """
        )
