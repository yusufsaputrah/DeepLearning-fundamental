from google_play_scraper import Sort, reviews
import pandas as pd
import time

def scrape_reviews(app_id, lang='id', country='id', num_reviews=12000):
    print(f"Scraping reviews for {app_id}...")
    
    result, continuation_token = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=num_reviews,
        filter_score_with=None
    )
    
    print(f"Successfully scraped {len(result)} reviews.")
    
    # Convert to DataFrame
    df = pd.DataFrame(result)
    
    # Select relevant columns: 'content', 'score'
    df = df[['content', 'score', 'at']]
    
    # Map score to sentiment
    # 1, 2 -> Negatif
    # 3 -> Netral
    # 4, 5 -> Positif
    def map_sentiment(score):
        if score <= 2:
            return 'Negatif'
        elif score == 3:
            return 'Netral'
        else:
            return 'Positif'
            
    df['sentiment'] = df['score'].apply(map_sentiment)
    
    return df

if __name__ == "__main__":
    app_package = 'com.gojek.app'
    
    start_time = time.time()
    
    # We aim for > 10000 reviews
    df_reviews = scrape_reviews(app_package, num_reviews=12000)
    
    print("Class distribution:")
    print(df_reviews['sentiment'].value_counts())
    
    output_file = 'dataset_reviews.csv'
    df_reviews.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"Data saved to {output_file}. Process took {end_time - start_time:.2f} seconds.")
