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

        self.cb_books = None
        self.cosine_sim = None
        self.cb_indices = None

        self.pivot_matrix = None
        self.knn_model = None

    def initialize(
        self,
        cb_subset_size=20000,
        min_user_ratings=50,
        min_book_ratings=20,
    ):
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

            author, publisher = 'N/A', 'N/A'
            matched = self.books[self.books['book_title'] == rec_title]
            if not matched.empty:
                author = matched.iloc[0]['book_author']
                publisher = matched.iloc[0]['publisher']

            recommendations.append({
                'book_title': rec_title,
                'book_author': author,
                'publisher': publisher,
                'similarity_score': sim,
            })

        return pd.DataFrame(recommendations)

    def get_user_history(self, user_id: int):
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_history = user_ratings.merge(self.books, on='isbn')[
            ['book_title', 'book_author', 'book_rating']
        ]
        return user_history.sort_values(by='book_rating', ascending=False)

    def recommend_for_user(self, user_id: int, top_n: int = 5):
        history = self.get_user_history(user_id)
        if history.empty:
            return None, 'No history found for this user.'

        top_books = history[history['book_title'].isin(self.pivot_matrix.index)]

        if top_books.empty:
            return (
                None,
                'User rated books, but none are in the filtered matrix.',
            )

        favorite_book = top_books.iloc[0]['book_title']
        recommendations = self.recommend_collaborative(
            favorite_book, top_n=top_n
        )

        return favorite_book, recommendations


# ==========================================
# 2. Modern Streamlit UI Configuration
# ==========================================
st.set_page_config(
    page_title='📚 Book Recommender Pro', page_icon='📖', layout='wide'
)

# Inject Custom CSS for Card UI Styling
st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    
    .rec-card {
        background: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 14px;
        transition: transform 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
    }
    .rec-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .rec-meta {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 10px;
    }
    .score-badge {
        background-color: #eff6ff;
        color: #2563eb;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""",
    unsafe_allow_html=True,
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


# Header
st.title('📚 Intelligent Book Recommender Pro')
st.caption('A dual-engine platform utilizing **Collaborative Filtering (KNN)** and **Content-Based Metadata Matching (CBF)**')

with st.spinner('🚀 Training recommendation models and loading data...'):
    engine = get_trained_engine()

# Summary Metrics Dashboard
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label='📖 Total Books', value=f'{len(engine.books):,}')
with col_m2:
    st.metric(label='⭐ Total Ratings', value=f'{len(engine.ratings):,}')
with col_m3:
    st.metric(
        label='👥 Active Users (CF)', value=f'{len(engine.pivot_matrix.columns):,}'
    )
with col_m4:
    st.metric(
        label='🎯 Filtered Book Catalog', value=f'{len(engine.pivot_matrix.index):,}'
    )

st.divider()

# Sidebar Controls
st.sidebar.header('⚙️ Control Panel')
top_n = st.sidebar.slider(
    'Recommendations Count (Top N):', min_value=1, max_value=10, value=5
)


# Helper function to render recommendation cards
def render_rec_cards(df_recs):
    if df_recs is None or df_recs.empty:
        st.warning('No matching recommendations found.')
        return

    for _, row in df_recs.iterrows():
        title = row.get('book_title', 'Unknown Title')
        author = row.get('book_author', 'N/A')
        publisher = row.get('publisher', 'N/A')
        score = row.get('similarity_score', 0.0)

        st.markdown(
            f"""
            <div class="rec-card">
                <div class="rec-title">📖 {title}</div>
                <div class="rec-meta">👤 Author: <b>{author}</b> &nbsp;|&nbsp; 🏢 Publisher: <b>{publisher}</b></div>
                <div class="score-badge">Similarity Score: {score}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Navigation Tabs
tab1, tab2 = st.tabs(
    ['👤 Personalized Recommendations (User Portal)', '🔍 Search by Book Title (Book Portal)']
)

# ==========================================
# TAB 1: User Personalization
# ==========================================
with tab1:
    st.markdown('### 👤 Personalized Recommendations based on User History')

    available_users = sorted(list(engine.pivot_matrix.columns))
    selected_user_id = st.selectbox(
        'Select a User ID:', available_users, key='user_select'
    )

    if st.button('✨ Fetch User Recommendations', type='primary', key='btn_user'):
        col_hist, col_rec = st.columns([1, 1.2])

        with col_hist:
            st.markdown(
                f'#### 📜 Top Rated Books by User `{selected_user_id}`'
            )
            user_history = engine.get_user_history(selected_user_id)
            if not user_history.empty:
                st.dataframe(
                    user_history.head(8),
                    use_container_width=True,
                    height=380,
                )
            else:
                st.info('No rating history available for this user.')

        with col_rec:
            fav_book, cf_recs = engine.recommend_for_user(
                selected_user_id, top_n=top_n
            )
            if cf_recs is not None and not cf_recs.empty:
                st.markdown(
                    f"#### 💡 Recommended based on top-rated book **'{fav_book}'**:"
                )
                render_rec_cards(cf_recs)
            else:
                st.warning('Unable to generate collaborative recommendations for this user.')

# ==========================================
# TAB 2: Book Title Search
# ==========================================
with tab2:
    st.markdown('### 🔍 Find Similar Books by Title')

    available_books = sorted(list(engine.pivot_matrix.index))
    selected_book = st.selectbox(
        'Type or select a book title:', available_books, key='book_select'
    )

    algorithm = st.radio(
        'Select Recommendation Engine:',
        ('Collaborative Filtering', 'Content-Based Filtering', 'Dual-Engine Comparison'),
        horizontal=True,
    )

    if st.button('✨ Generate Recommendations', type='primary', key='btn_book'):
        st.divider()

        if algorithm in ['Content-Based Filtering', 'Dual-Engine Comparison']:
            st.markdown('#### 📖 Content-Based Recommendations (Metadata Matching)')
            cb_results = engine.recommend_content_based(selected_book, top_n)
            render_rec_cards(cb_results)

        if algorithm in ['Collaborative Filtering', 'Dual-Engine Comparison']:
            st.markdown('#### 👥 Collaborative Filtering Recommendations (KNN User Behavior)')
            cf_results = engine.recommend_collaborative(selected_book, top_n)
            render_rec_cards(cf_results)