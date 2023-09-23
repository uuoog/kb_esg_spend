from konlpy.tag import Komoran
from collections import Counter, defaultdict
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import streamlit as st
import pandas as pd
import numpy as np
import mpld3
import time
import re

# ======================================================================================================================
# streamlit 설정
# ======================================================================================================================
# strealit font 설정 (구글 font만 가능)
font = "Noto Sans Korean"

# strealit 배경색 설정
backgroundColor = "#F0F0F0"
# st.set_page_config(layout="wide")
# ======================================================================================================================
# data 선언
# ======================================================================================================================
influence_df = pd.read_csv("./data/brand_embedding_label_df.csv")

# ======================================================================================================================
# 변수 선언
# ======================================================================================================================
komoran = Komoran(userdic="./data/user.dic")
brand = list(influence_df["브랜드 이름"].unique())
brand = ["브랜드 선택"] + brand
e_word_count_dict = defaultdict(Counter)
s_word_count_dict = defaultdict(Counter)
g_word_count_dict = defaultdict(Counter)

color_palette = ["#ffa505", "#ffb805", "#ffc905", "#ffe505", "#fffb05"]
# ======================================================================================================================
# 함수 선언
# ======================================================================================================================

def remove_special_characters(text):
    # 정규 표현식을 사용하여 특수문자 제거
    text = re.sub(r'[^가-힣a-zA-Z\s]', ' ', text)
    return text

@st.cache_data
def tokenize(x):
    try:
        tokens = komoran.pos(x)
        return tokens
    except Exception as e:
        print(e, x)
        return None

def extract_nouns(tokens):
    return [text for text, tag in tokens if tag in ("NNP", "NNG")]

# def count_words(nouns):
#     brand_name_nouns = set(nouns).intersection(brand)
#     for brand_name in brand_name_nouns:
#         c = Counter([x for x in nouns if x != brand_name])
#         word_count_dict[brand_name] += c

e_mask_image = np.array(Image.open("./img/word_cloud_e.png"))
s_mask_image = np.array(Image.open("./img/word_cloud_s.png"))
g_mask_image = np.array(Image.open("./img/word_cloud_g.png"))

def visualize_wordcloud(word_count, color, mask_image):
    wordcloud = WordCloud(
        font_path="./fonts/NanumGothicCoding.ttf",
        width=1000,
        height=400,
        scale=2.0,
        background_color='white',
        mask=mask_image,
        colormap=color,
        max_font_size=150,
    ).generate_from_frequencies(word_count)
    wordcloud_image = wordcloud.to_image()
    return wordcloud_image

# ======================================================================================================================
# 실행부
# ======================================================================================================================
influence_df["content"] = influence_df["본문"]
influence_df['content'] = influence_df['content'].str.replace('\n', ' ')
influence_df['content'] = influence_df['content'].apply(remove_special_characters)

tqdm.pandas()
influence_df['tokenized'] = influence_df['content'].progress_apply(lambda x: tokenize(x))

influence_df["nouns"] = influence_df["tokenized"].progress_apply(lambda x: extract_nouns(x))


e_content_nouns_series = influence_df[influence_df["지표"] == "환경"]["nouns"]
s_content_nouns_series = influence_df[influence_df["지표"] == "사회"]["nouns"]
g_content_nouns_series = influence_df[influence_df["지표"] == "지배구조"]["nouns"]


if not e_content_nouns_series.empty:
    for content_nouns in tqdm(e_content_nouns_series):
        # count_words(content_nouns)
        brand_name_nouns = set(content_nouns).intersection(brand)
        for brand_name in brand_name_nouns:
            c = Counter([x for x in content_nouns if x != brand_name])
            e_word_count_dict[brand_name] += c

if not s_content_nouns_series.empty:
    for content_nouns in tqdm(s_content_nouns_series):
        brand_name_nouns = set(content_nouns).intersection(brand)
        for brand_name in brand_name_nouns:
            c = Counter([x for x in content_nouns if x != brand_name])
            s_word_count_dict[brand_name] += c

if not g_content_nouns_series.empty:
    for content_nouns in tqdm(g_content_nouns_series):
        brand_name_nouns = set(content_nouns).intersection(brand)
        for brand_name in brand_name_nouns:
            c = Counter([x for x in content_nouns if x != brand_name])
            g_word_count_dict[brand_name] += c

# ======================================================================================================================
with st.form("Wordcloud"):
    st.title('wordcloud')
    customer_list = ["차국민", "라국민", "허리브", "정국민", "현국민", "강리브"]
    selected_brand = st.selectbox('', brand,  placeholder='Select...')
    submitted = st.form_submit_button("조회")

    if submitted:
        if selected_brand == "브랜드 선택":
            st.write("브랜드를 선택해주세요")
        elif selected_brand not in brand:
            st.write("해당 브랜드는 조회되지 않습니다.")
        else:
            if not e_content_nouns_series.empty:
                e_wordcloud_image = visualize_wordcloud(e_word_count_dict[selected_brand], "YlOrBr", e_mask_image)
            if not s_content_nouns_series.empty:
                s_wordcloud_image = visualize_wordcloud(s_word_count_dict[selected_brand], "YlOrBr", s_mask_image)
            if not g_content_nouns_series.empty:
                g_wordcloud_image = visualize_wordcloud(g_word_count_dict[selected_brand], "YlOrBr", g_mask_image)

            col1, col2, col3 = st.columns(3)

            with st.spinner("wordcloud를 구성 중 입니다."):
                time.sleep(1)
                with col1:
                    if not e_content_nouns_series.empty:
                        st.image(e_wordcloud_image, caption='환경')
                with col2:
                    if not s_content_nouns_series.empty:
                        st.image(s_wordcloud_image, caption='사회')
                with col3:
                    if not g_content_nouns_series.empty:
                        st.image(g_wordcloud_image, caption='지배구조')

# ======================================================================================================================
# selected_brand = st.selectbox('브랜드 선택', brand,  placeholder='Select...')
# # selected_brand = st.text_input("브랜드 이름을 입력하세요:",  placeholder='브랜드 입력(영어는 대문자입력)')
#
# if selected_brand == "선택안함":
#     st.write("브랜드를 입력해 주세요.")
# elif selected_brand not in brand:
#     st.write("해당 브랜드는 조회되지 않습니다.")
# else:
#     if not e_content_nouns_series.empty:
#         e_wordcloud_image = visualize_wordcloud(e_word_count_dict[selected_brand], "YlOrBr", e_mask_image)
#     if not s_content_nouns_series.empty:
#         s_wordcloud_image = visualize_wordcloud(s_word_count_dict[selected_brand], "YlOrBr", s_mask_image)
#     if not g_content_nouns_series.empty:
#         g_wordcloud_image = visualize_wordcloud(g_word_count_dict[selected_brand], "YlOrBr", g_mask_image)
#
#     col1, col2, col3 = st.columns(3)
#
#     with st.spinner("wordcloud를 구성 중 입니다."):
#         time.sleep(1)
#         with col1:
#             if not e_content_nouns_series.empty:
#                 st.image(e_wordcloud_image, caption='환경')
#         with col2:
#             if not s_content_nouns_series.empty:
#                 st.image(s_wordcloud_image, caption='사회')
#         with col3:
#             if not g_content_nouns_series.empty:
#                 st.image(g_wordcloud_image, caption='지배구조')

    # with st.spinner("wordcloud를 구성 중 입니다."):
    #     time.sleep(1)
    #     if not e_content_nouns_series.empty:
    #         st.image(e_wordcloud_image, caption='환경')
    #     if not s_content_nouns_series.empty:
    #         st.image(s_wordcloud_image, caption='사회')
    #     if not g_content_nouns_series.empty:
    #         st.image(g_wordcloud_image, caption='지배구조')
