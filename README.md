# Bank Credit and GDP Growth in India: Interactive Dashboard

An interactive companion to the study of how bank credit growth relates to GDP growth in India, 1971-72 to 2025-26.

## What it shows

| Tab | What you can do |
|---|---|
| Overview | Headline indicators, credit and GDP growth over time, credit/GDP and deposits/GDP, credit-GDP growth gap |
| Relationship | Scatter with fitted line and live correlation statistics, rolling correlation, GDP growth by credit-growth bucket, regime summary |
| Lead-lag | Cross-correlations at adjustable leads and lags (growth rates or HP-filtered cycles), Granger causality tests |
| Episodes | Boom, crisis, stress and revival episodes, or any custom window |
| Composition and stress | Sector shares and growth of credit, comparison with investment and consumption growth, NPAs against credit growth |
| Geography of credit | Rural, semi-urban, urban and metropolitan credit and deposits, treemap, shares and credit-deposit ratios by year |
| Simulator | Credit/GDP projection with scenario grid, and a Monte Carlo model of credit responding to a GDP growth path |
| Data and notes | Download the datasets, sources and adjustments |

Sidebar controls (year range, credit measure, masking of 2020-21) apply across the analysis tabs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, choose **New app**, and select the repository.
3. Set the main file path to `app.py` and deploy.

## Structure

```
app.py              dashboard
utils.py            data loading and statistical helpers
data/               datasets (CSV)
requirements.txt
.streamlit/config.toml
```

## Data

RBI Handbook of Statistics on the Indian Economy 2025-26 (Tables 232, 41, 233, 235, 236, 227, 42, 50, 47, 4) and MoSPI national accounts for GDP growth before 2022-23. Credit growth for 2023-24 and 2024-25 is adjusted for the 2023 bank merger. 2025-26 figures are provisional.
