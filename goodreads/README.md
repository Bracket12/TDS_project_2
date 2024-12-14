# Data Analysis Report

## 1. Data Overview

The dataset contains information about 10,000 books, with various attributes including the following:

- **Identifiers**: `book_id`, `goodreads_book_id`, `best_book_id`, and `work_id`.
- **Book Details**: `books_count`, `isbn`, `isbn13`, `authors`, `original_publication_year`, `original_title`, `title`, `language_code`, `average_rating`, `ratings_count`, and more.
- **Ratings**: The dataset includes detailed ratings breakdowns (`ratings_1` to `ratings_5`) and review counts (`work_text_reviews_count`).

## 2. Analysis Conducted

### Summary Statistics
A summary of the statistics was generated to understand the distribution of key metrics in the dataset. Key observations include:
- The average rating across books is approximately **4.18**.
- The maximum number of ratings received by any book is **4,780,653**.
- There are **4664** unique authors represented in the dataset.

### Missing Values
Missing values were identified, with **10.84%** of the entries lacking a `language_code`, and **7.00%** missing `isbn`. The analysis shows that most of the critical fields have no missing values.

### Correlation Analysis
A correlation matrix was created to understand relationships between different variables. For instance, `ratings_count` and `work_ratings_count` are strongly correlated, indicating that books with more ratings tend to have more work ratings as well.

### Outlier Detection
Outliers were detected based on various attributes, with specific books showing unusual values for ratings and counts, indicating they may have received atypically high engagement.

### Clustering and PCA
K-means clustering was applied to categorize books into distinct groups based on various features, yielding **4 clusters**. PCA was also conducted, revealing that the first two principal components explain a significant portion of the variance in the dataset.

### Hierarchical Clustering
Hierarchical clustering was performed to visualize the relationships between books based on their features, which can be useful for understanding similarities and differences.

## 3. Key Insights

- **High Ratings and Count Correlation**: Books with higher average ratings tend to have a higher count of ratings, suggesting that quality and popularity are interconnected.
- **Author Proliferation**: The dataset reveals a diverse array of authors, with some authors being more prolific than others, which could be explored further to identify trends in author output and reader engagement.
- **Language Distribution**: The variety of languages present could indicate a broad market for translated works, suggesting potential for cross-cultural engagement.

### Visualizations
The following visualizations support the insights derived from the analysis:

#### Average Rating by Year
![average_rating_by_year](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAEsASwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD0CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//2Q==)

#### Correlation Heatmap
![correlation_heatmap](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAEsASwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRA