# Data Analysis Report

## 1. Data Overview
The dataset consists of 2,363 records across multiple countries covering various life quality indicators over the years. Key attributes include:
- **Country Name**: The name of the country (165 unique countries).
- **Year**: The year of data collection (ranging from 2005 to 2023).
- **Life Ladder**: A self-reported measure of life satisfaction.
- **Log GDP per Capita**: The logarithm of GDP per capita.
- **Social Support**: A measure of perceived support from family and friends.
- **Healthy Life Expectancy at Birth**: Average number of years a newborn is expected to live in good health.
- **Freedom to Make Life Choices**: A perception-based metric.
- **Generosity**: A measure of charitable giving.
- **Perceptions of Corruption**: A measure of how corrupt the country is perceived to be.
- **Positive and Negative Affect**: Emotional well-being indicators.

### Summary Statistics
| Variable                            | Count | Mean      | Std Dev   | Min   | Max   |
|-------------------------------------|-------|-----------|-----------|-------|-------|
| Year                                | 2363  | 2014.76   | 5.06      | 2005  | 2023  |
| Life Ladder                         | 2363  | 5.48      | 1.13      | 1.28  | 8.02  |
| Log GDP per Capita                 | 2335  | 9.40      | 1.15      | 5.53  | 11.68 |
| Social Support                      | 2350  | 0.81      | 0.12      | 0.23  | 0.99  |
| Healthy Life Expectancy at Birth    | 2300  | 63.40     | 6.84      | 6.72  | 74.60 |
| Freedom to Make Life Choices        | 2327  | 0.75      | 0.14      | 0.23  | 0.99  |
| Generosity                          | 2282  | 0.0001    | 0.16      | -0.34 | 0.70  |
| Perceptions of Corruption           | 2238  | 0.74      | 0.18      | 0.04  | 0.98  |
| Positive Affect                     | 2339  | 0.65      | 0.11      | 0.18  | 0.88  |
| Negative Affect                     | 2347  | 0.27      | 0.09      | 0.08  | 0.71  |

## 2. Analysis Conducted
- **Correlation Analysis**: Evaluated relationships between various indicators to understand how they interrelate.
- **Missing Values Assessment**: Identified missing values in several columns to assess data quality.
- **Outlier Detection**: Analyzed data for outliers in key metrics that could skew results.
- **K-Means Clustering**: Grouped countries into clusters based on their scores across the various metrics.
- **Principal Component Analysis (PCA)**: Reduced dimensionality of the dataset to identify major factors contributing to life satisfaction.

## 3. Insights Discovered
- **Strong Correlations**: The strongest correlation was found between **Log GDP per Capita** and **Life Ladder** (0.78), indicating that wealthier countries tend to report higher life satisfaction.
- **Social Support Impact**: There is a significant positive correlation (0.72) between **Social Support** and **Life Ladder**, suggesting that perceived support from family and friends plays a crucial role in life satisfaction.
- **Negative Correlation with Corruption**: A notable negative correlation exists between **Perceptions of Corruption** and the **Life Ladder** score (-0.43), highlighting that countries perceived as more corrupt tend to have lower life satisfaction metrics.
- **PCA Results**: The first two principal components explained 51.4% of the variance in the dataset, indicating that a significant portion of the data structure can be captured with these components.

## 4. Implications of Findings
- **Policy Recommendations**: Focus on enhancing social support systems may lead to increased life satisfaction, particularly in countries where social support is lacking.
- **Economic Strategies**: Initiatives aimed at economic growth, particularly through GDP improvement, should be prioritized as they correlate strongly with life satisfaction.
- **Corruption Focus**: Efforts to reduce corruption could improve citizens' perceptions and overall happiness in affected countries.
- **Monitoring and Evaluation**: Continuous monitoring of these metrics can help in tracking the effectiveness of policies aimed at improving life quality.

---

### Visualizations
1. **Correlation Heatmap**
   ![correlation_heatmap](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAEsASwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ