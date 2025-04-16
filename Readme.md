# Anime Recommendation System

## What is a Recommendation System?

A recommendation system is an algorithm that suggests relevant items to users based on various factors. There are three main types:

1. **Content-Based Filtering**: Recommends items similar to what a user has liked before, based on item features
2. **Collaborative Filtering**: Recommends items based on preferences of users with similar tastes
3. **Hybrid Systems**: Combines both approaches

## Requirements
- Python 3.6+

## Setup

1. Clone this repository
2. Set up the virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Make sure the anime dataset (`anime.csv`) is in the `dataset` folder

## Usage

### Jupyter Notebook

The main implementation is in the Jupyter notebook `model.ipynb`. To run it:

1. Activate the virtual environment
2. Start Jupyter:
   ```
   jupyter notebook
   ```
3. Open `model.ipynb`
4. Run all cells to train the model and use the recommendation system

<!-- ### Python Script

You can also use the recommendation system through the provided test script:

1. First, run the notebook to train and save the model
2. Then run the test script:
   ```
   python test_recommender.py
   ``` -->


## How It Works

1. **Data Loading**: The system loads anime data from the CSV file, focusing on specific columns like genres, studios, themes, etc.
2. **Preprocessing**: Missing values are handled, and text data is cleaned.
3. **Feature Engineering**: Text features are combined to create a rich representation of each anime.
4. **TF-IDF Vectorization**: Text data is converted to numerical features using TF-IDF.
5. **Similarity Calculation**: Cosine similarity is used to find how similar anime are to each other.
6. **Recommendation**: When a user inputs an anime they like, the system finds the most similar anime based on content features.

## Columns Used

The system uses the following columns from the anime dataset:
- `anime_id`: Unique identifier for each anime
- `image_url`: URL to the anime's image
- `describe`: Description/synopsis of the anime
- `english_name`: English title of the anime
- `score`: Average user rating
- `genres`: Categories the anime belongs to
- `producers`: Companies that produced the anime
- `studios`: Studios that created the anime
- `themes`: Thematic elements in the anime
- `rank`: Popularity ranking
- `episodes`: Number of episodes
- `popularity`: Popularity metric
- `favorites`: Number of users who favorited the anime

## Future Improvements

- Add collaborative filtering to consider user preferences
- Implement more advanced NLP techniques for better text analysis
- Create a web interface for a more user-friendly experience
- Add explanations for why each anime is recommended
- Include more features like anime type, episodes, etc.

## License

This project is open source and available for educational purposes.
