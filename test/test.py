"""
Test script for the Anime Recommendation System.
"""

# Import necessary libraries
import dill

# Step 1: Load the saved model
print("Loading the anime recommendation model...")
try:
    dilld_model = dill.load(open("model/anime_recommender.pkl", "rb"))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'anime_recommender.pkl' not found!")
    exit()
except Exception as e:
    print(f"Error in loading the model : {e}")
    exit()


# Step 2: Access the recommender component from our model
# Our model is a pipeline with an 'AnimeRecommender' step that does the actual work
recommender = dilld_model.named_steps['AnimeRecommender']

### Step 3: 

# Title
print("\n" + "-"*50)
print("Welcome to the Anime Recommendation System!")
print("-"*50)

# Get user input for AnimeRecommender

while True:
    print("\n" + "-"*50)
    print("1. Get Recommendations")
    print("2. Get Recommendations By Genre")
    print("3. Get Top Rated Animes")
    print("4. Get Popular Animes")
    print("5. Search for an Anime")
    print("6. Exit")
    print("-"*50)


    choice = input("\n Please enter your choice: ")

    if choice not in ["1", "2", "3", "4", "5", "6"]:
        print("\nInvalid choice. Please try again.")
        continue

    if choice == "1":
        anime_name = input("\nEnter the name of an anime you like: ")
        try:
            n = int(input("\nEnter the number of recommendations you want: "))
        except ValueError:
            print("\nInvalid input. Please enter a number.")
            continue

        recommendations = recommender.get_recommendations(anime_name, n)

        if recommendations.empty:
            continue
        else:
            print(f"\nTop {n} Recommendations similar to '{anime_name}':\n")

        # Loop through each recommendation
        for i in range(len(recommendations)):
            # Get the data for the current recommendation
            anime = recommendations.iloc[i]

            # Print the information in a nice format
            print(f"{i+1}. Name: {anime['english_name']}")
            print(f"   Genres: {anime['genres']}")
            print(f"   Score: {anime['score']}")
            print()



    elif choice == "2":
        genre = input("\nEnter the genre of anime you like: ")
        try:
            n = int(input("\nEnter the number of recommendations you want: "))
        except ValueError:
            print("\nInvalid input. Please enter a number.")
            continue

        recommendations = recommender.get_recommendations_by_genre(genre, n)
            # If the anime wasn't found, the recommender returns an empty DataFrame
        if recommendations.empty:
            continue
        else:
            print(f"\nTop {n} Recommendations similar to '{genre}':\n")

        # Loop through each recommendation
        for i in range(len(recommendations)):
            # Get the data for the current recommendation
            anime = recommendations.iloc[i]

            # Print the information in a nice format
            print(f"{i+1}. Name: {anime['english_name']}")
            print(f"   Genres: {anime['genres']}")
            print(f"   Score: {anime['score']}")
            print()

    elif choice == "3":
        try:
            n = int(input("\nEnter the number of Top Rated Animes you want: "))
        except ValueError:
            print("\nInvalid input. Please enter a number.")
            continue

        get_top_rated_animes = recommender.get_top_rated_anime(n)

        print(f"\n Top {n} Rated Animes: \n")

        # Loop through each recommendation
        for i in range(len(get_top_rated_animes)):
            # Get the data for the current recommendation
            anime = get_top_rated_animes.iloc[i]

            # Print the information in a nice format
            print(f"{i+1}. Name: {anime['english_name']}")
            print(f"   Genres: {anime['genres']}")
            print(f"   Score: {anime['score']}")
            print()

    elif choice == "4":
        try:
            n = int(input("\nEnter the number of recommendations you want: "))
        except ValueError:
            print("\nInvalid input. Please enter a number.")
            continue

        get_top_rated_animes = recommender.get_top_rated_anime(n)

        print(f"\n Top {n} Popular Animes: \n")

        # Loop through each recommendation
        for i in range(len(get_top_rated_animes)):
            # Get the data for the current recommendation
            anime = get_top_rated_animes.iloc[i]

            # Print the information in a nice format
            print(f"{i+1}. Name: {anime['english_name']}")
            print(f"   Genres: {anime['genres']}")
            print(f"   Score: {anime['score']}")
            print()


    elif choice == "5":
        anime_name = input("\n Search for an anime : ")
        try:
            n = int(input("\nEnter the number of recommendations you want: "))
        except ValueError:
            print("\nInvalid input. Please enter a number.")
            continue
        recommendations = recommender.search_anime(anime_name, n)
        if recommendations.empty:
            print(f"\nNo Anime found for '{anime_name}'. Please check the spelling or try another anime.")
        else:
            print(f"\nTop {n} Animes similar to '{anime_name}':\n")

        # Loop through each recommendation
        for i in range(len(recommendations)):
            # Get the data for the current recommendation
            anime = recommendations.iloc[i]

            # Print the information in a nice format
            print(f"{i+1}. Name: {anime['english_name']}")
            print(f"   Genres: {anime['genres']}")
            print(f"   Score: {anime['score']}")
            print()


    elif choice == "6":
        print("\nThank you for using the Anime Recommendation System!")
        exit()