import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import streamlit as st


# ==========================================
# 1. Core Recommender Engine Class
# ==========================================
class BookRecommenderEngine:

    def __init__(self, books_path: str, ratings_path: str):
        self.books_path = books_path
        self.ratings_path = ratings_path

        self.books = None
        self.ratings = None

        # Content-Based Attributes
        self.cb_books = None
        self.cosine_sim = None
        self.cb_indices = None

        # Collaborative Filtering Attributes
        self.pivot_matrix = None
        self.knn_model = None

    def initialize(
        self,
        cb_subset_size=20000,
        min_user_ratings=50,
        min_book_ratings=20,
    ):
        # 1. Load and clean dataset
        self.books = pd.read_csv(
            self.books_path,
            encoding='latin-1',
            on_bad_lines='skip',
            low_memory=False,
        )
        self.ratings = pd.read_csv(
            self.ratings_path,
            encoding='latin-1',
            on_bad_lines='skip',
            low_memory=False,
        )

        self.books.columns = (
            self.books.columns.str.strip().str.lower().str.replace('-', '_')
        )
        self.ratings.columns = (
            self.ratings.columns.str.strip().str.lower().str.replace('-', '_')
        )

        image_cols = ['image_url_s', 'image_url_m', 'image_url_l']
        self.books.drop(columns=image_cols, errors='ignore', inplace=True)
        self.books.fillna('', inplace=True)

        # 2. Train Content-Based Model
        self.cb_books = (
            self.books.iloc[:cb_subset_size].copy().reset_index(drop=True)
        )
        self.cb_books['metadata'] = (
            self.cb_books['book_title'].astype(str)
            + ' '
            + self.cb_books['book_author'].astype(str)
            + ' '
            + self.cb_books['publisher'].astype(str)
        )

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.cb_books['metadata'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.cb_indices = pd.Series(
            self.cb_books.index, index=self.cb_books['book_title']
        ).drop_duplicates()

        # 3. Train Collaborative Filtering (KNN) Model
        df = self.ratings.merge(self.books, on='isbn')

        active_users = df['user_id'].value_counts()[
            lambda x: x >= min_user_ratings
        ].index
        df_filtered = df[df['user_id'].isin(active_users)]

        popular_books = df_filtered['book_title'].value_counts()[
            lambda x: x >= min_book_ratings
        ].index
        df_filtered = df_filtered[
            df_filtered['book_title'].isin(popular_books)
        ]

        self.pivot_matrix = df_filtered.pivot_table(
            index='book_title', columns='user_id', values='book_rating'
        ).fillna(0)

        sparse_matrix = csr_matrix(self.pivot_matrix.values)
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_model.fit(sparse_matrix)

    def recommend_content_based(self, title: str, top_n: int = 5):
        if self.cb_indices is None or title not in self.cb_indices:
            return None

        idx = self.cb_indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
            1 : top_n + 1
        ]
        book_indices = [i[0] for i in sim_scores]

        results = self.cb_books.iloc[book_indices][
            ['book_title', 'book_author', 'publisher']
        ].copy()
        results['similarity_score'] = [round(i[1], 4) for i in sim_scores]
        return results

    def recommend_collaborative(self, title: str, top_n: int = 5):
        if (
            self.pivot_matrix is None
            or title not in self.pivot_matrix.index
        ):
            return None

        book_idx = self.pivot_matrix.index.get_loc(title)
        if isinstance(book_idx, (slice, np.ndarray)):
            book_idx = book_idx[0]

        distances, indices = self.knn_model.kneighbors(
            self.pivot_matrix.iloc[book_idx, :].values.reshape(1, -1),
            n_neighbors=top_n + 1,
        )

        recommendations = []
        for i in range(1, len(distances.flatten())):
            rec_title = self.pivot_matrix.index[indices.flatten()[i]]
            sim = round(1 - distances.flatten()[i], 4)
            recommendations.append(
                {'book_title': rec_title, 'similarity_score': sim}
            )

        return pd.DataFrame(recommendations)

    def get_user_history(self, user_id: int):
        """Fetch rating history for a given user ID."""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_history = user_ratings.merge(self.books, on='isbn')[
            ['book_title', 'book_author', 'book_rating']
        ]
        return user_history.sort_values(by='book_rating', ascending=False)

    def recommend_for_user(self, user_id: int, top_n: int = 5):
        """Generate personalized recommendations for a specific User ID."""
        history = self.get_user_history(user_id)
        if history.empty:
            return None, 'No history found for this user.'

        # Find the highest-rated book by the user that exists in our CF pivot matrix
        top_books = history[history['book_title'].isin(self.pivot_matrix.index)]

        if top_books.empty:
            return (
                None,
                'User rated books, but none are in the filtered Matrix.',
            )

        favorite_book = top_books.iloc[0]['book_title']
        recommendations = self.recommend_collaborative(
            favorite_book, top_n=top_n
        )

        return favorite_book, recommendations


# ==========================================
# 2. Streamlit Web Interface
# ==========================================
st.set_page_config(
    page_title='📚 Book Recommendation System',
    page_icon='📚',
    layout='wide',
)


@st.cache_resource
def get_trained_engine():
    engine = BookRecommenderEngine(
        books_path='Books.csv', ratings_path='Ratings.csv'
    )
    engine.initialize(
        cb_subset_size=20000, min_user_ratings=50, min_book_ratings=20
    )
    return engine


st.title('📚 Intelligent Book Recommendation System')
st.markdown(
    'A dual-engine recommendation platform supporting **Book Title Search** & **Personalized User ID Recommendations**.'
)

with st.spinner('🚀 Training models and loading dataset...'):
    engine = get_trained_engine()

# Sidebar Controls
st.sidebar.header('⚙️ Recommendation Settings')

# Select Mode
recommendation_mode = st.sidebar.radio(
    'Select Recommendation Mode:',
    ('👤 Personalized by User ID', '📖 Search by Book Title'),
)

top_n = st.sidebar.slider(
    'Number of Recommendations (Top N):', min_value=1, max_value=10, value=5
)


# ==========================================
# MODE 1: Personalized Recommendations by User ID
# ==========================================
if recommendation_mode == '👤 Personalized by User ID':
    st.subheader('👤 User ID Portal')

    available_users = sorted(list(engine.pivot_matrix.columns))
    selected_user_id = st.selectbox('Select a User ID:', available_users)

    if st.button('✨ Fetch User Profile & Recommendations', type='primary'):
        st.divider()

        # Display User Rating History
        st.write(f'### 📜 Rating History for User ID: `{selected_user_id}`')
        user_history = engine.get_user_history(selected_user_id)

        if not user_history.empty:
            st.dataframe(user_history.head(10), use_container_width=True)

            # Generate Recommendations based on User's Favorite Book
            fav_book, cf_recs = engine.recommend_for_user(
                selected_user_id, top_n=top_n
            )

            if cf_recs is not None and not cf_recs.empty:
                st.success(
                    f"💡 Based on User {selected_user_id}'s top-rated book **'{fav_book}'**, here are the recommended books:"
                )
                st.dataframe(cf_recs, use_container_width=True)
            else:
                st.warning(
                    'Could not generate collaborative recommendations for this user.'
                )
        else:
            st.warning('No rating history found for this user.')


# ==========================================
# MODE 2: Recommendations by Book Title
# ==========================================
else:
    st.subheader('🔍 Book Title Search Portal')

    available_books = sorted(list(engine.pivot_matrix.index))
    selected_book = st.selectbox('Type or select a book title:', available_books)

    algorithm = st.sidebar.radio(
        'Select Algorithm:', ('Both', 'Content-Based', 'Collaborative Filtering')
    )

    if st.button('✨ Generate Recommendations', type='primary'):
        st.divider()
        st.info(f"Generating results for **'{selected_book}'**:")

        if algorithm in ['Both', 'Content-Based']:
            st.subheader('📖 Content-Based Filtering')
            cb_results = engine.recommend_content_based(selected_book, top_n)
            if cb_results is not None:
                st.dataframe(cb_results, use_container_width=True)
            else:
                st.warning('Book not found in Content-Based subset.')

        if algorithm in ['Both', 'Collaborative Filtering']:
            st.subheader('👥 Collaborative Filtering (KNN)')
            cf_results = engine.recommend_collaborative(selected_book, top_n)
            if cf_results is not None:
                st.dataframe(cf_results, use_container_width=True)
            else:
                st.warning('Insufficient rating data for Collaborative Filtering.')