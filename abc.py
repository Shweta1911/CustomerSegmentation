import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
df = pd.read_csv('OnlineRetail.csv')

# Convert implicit feedback to binary ratings (1 for interaction, 0 for no interaction)
df['Rating'] = (df['Quantity'] > 0).astype(int)

# Streamlit UI
st.title("Product ID Recommendation")
user_id = st.text_input(label="Customer ID")

if not user_id:
    st.warning("Input box is empty. Please enter something.")
else:
    # Create a Surprise reader and load the dataset
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[['CustomerID', 'StockCode', 'Rating']], reader)

    # Build the collaborative filtering model (FunkSVD)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)

    # Get items not interacted by the user
    items_not_rated = df.loc[~df['StockCode'].isin(df[df['CustomerID'] == int(user_id)]['StockCode']), 'StockCode'].unique()

    # Get predicted ratings for the items not rated by the user
    item_ratings = [(item_id, model.predict(int(user_id), item_id).est) for item_id in items_not_rated]

    # Sort the items by predicted ratings in descending order
    item_ratings = sorted(item_ratings, key=lambda x: x[1], reverse=True)

    # Display top N recommendations
    top_n = 5
    st.write(f"Top {top_n} Recommendations for User Using SVD {user_id}:")
    for i, (item_id, est_rating) in enumerate(item_ratings[:top_n], 1):
        st.write(f"{i}. Item {item_id}: Estimated Rating = {est_rating}")
