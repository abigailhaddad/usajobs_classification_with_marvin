import pandas as pd

def create_presence_df(df):
    """Create a DataFrame that indicates the presence (1) or absence (0) of each job title."""
    all_titles = df['job_title'].unique()
    presence_df = pd.DataFrame(columns=all_titles)
    
    for idx, row in df.iterrows():
        for title in all_titles:
            if title == row['job_title']:
                presence_df.at[idx, title] = 1
            else:
                presence_df.at[idx, title] = 0
    presence_df['Category'] = df['category']
    
    return presence_df

def get_title_proportions(df):
    """Get proportions of each title in each category."""
    presence_df = create_presence_df(df)
    proportions_df = presence_df.groupby('Category').mean() * 100
    return proportions_df

def get_top_n_titles(summary_df, n=5):
    """Get the top n job titles by percentage for each category and format it."""
    categories = summary_df.index.tolist()
    titles_list = []
    
    for _, row in summary_df.iterrows():
        top_titles = ", ".join([k for k, v in row.sort_values(ascending=False).head(n).to_dict().items() if v > 0])
        titles_list.append(top_titles)
        
    return pd.DataFrame({'Category': categories, 'Top Titles': titles_list})

def summarize_data(file_with_llm_markings):
    # Assuming df is loaded from somewhere
    df = pd.read_pickle(f"../data/{file_with_llm_markings}.pkl") # replace this with wherever you're loading your DataFrame
    
    summary_df = get_title_proportions(df)
    result_df = get_top_n_titles(summary_df)
    
    # Write to CSV
    result_df.to_csv('../results/top_titles_summary.csv', index=False)
    
    return result_df

