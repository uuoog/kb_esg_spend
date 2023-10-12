from collections import Counter, defaultdict
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.font_manager as fm
import streamlit as st
import pandas as pd
import numpy as np
import time
from st_pages import show_pages_from_config, add_page_title

# ======================================================================================================================
# streamlit 설정
# ======================================================================================================================
add_page_title(layout="wide")
show_pages_from_config()

# strealit font 설정 (구글 font만 가능)
font = "Noto Sans Korean"

# strealit 배경색 설정
backgroundColor = "#F0F0F0"

# plot font 설정
font_dirs = ['./fonts']
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)

fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'NanumGothicCoding'

# ======================================================================================================================
# data 선언
# ======================================================================================================================
spending_df = pd.read_csv("./data/base_data.csv", encoding="utf-8")
influence_df = pd.read_csv("./data/news_wordcloud_by_brand.csv")

influence_df['날짜'] = pd.to_datetime(influence_df['날짜'])
influence_df['날짜'] = influence_df['날짜'].dt.strftime('%Y-%m-%d')
influence_df["nouns"] = influence_df["nouns"].apply(eval)

# ======================================================================================================================
# 변수 선언
# ======================================================================================================================
color_palette = ["#ffa505", "#ffb805", "#ffc905", "#ffe505", "#fffb05"]
brand = list(influence_df["브랜드"].unique())
brand = ["브랜드 선택"] + brand
e_word_count_dict = defaultdict(Counter)
s_word_count_dict = defaultdict(Counter)
g_word_count_dict = defaultdict(Counter)
e_mask_image = np.array(Image.open("./img/word_cloud_e.png"))
s_mask_image = np.array(Image.open("./img/word_cloud_s.png"))
g_mask_image = np.array(Image.open("./img/word_cloud_g.png"))

# ======================================================================================================================
# 함수 선언
# ======================================================================================================================

# 유저 데이터 프레임 필터링
def filtered_spending_df(name):
    user_spending_df = spending_df[spending_df["이용 고객명"] == name]
    user_spending_df = user_spending_df.reset_index(drop=True)
    return user_spending_df

# 천 단위로 쉼표를 표시하는 함수 정의
def format_with_commas(value):
    return '{:,.0f}'.format(value)

# 브랜드 데이터 프레임 필터링
def filtered_brand_df(brand_name):
    brand_df = influence_df[influence_df["브랜드"] == brand_name]
    brand_df = brand_df.reset_index(drop=True)
    return brand_df

# 이용 고객님의 esg 소비 비중 그래프
def plot_esg_spending(brand_df):

    eco_count = (brand_df["esg_idx"] == "환경").sum()
    society_count = (brand_df["esg_idx"] == "사회").sum()
    governance_count = (brand_df["esg_idx"] == "지배구조").sum()

    total_count = brand_df["esg_idx"].count()
    eco_per = (eco_count / total_count) * 100
    society_per = (society_count / total_count) * 100
    governance_per = (governance_count / total_count) * 100

    fig, ax = plt.subplots(figsize=(10, 2))
    categories = ["ESG 소비 비중"]

    # 그래프 그리기
    bars1 = plt.barh(categories, eco_per, color=color_palette[4], label='환경')
    bars2 = plt.barh(categories, society_per, left=eco_per, color=color_palette[3], label='사회')
    bars3 = plt.barh(categories, governance_per, left=eco_per + society_per, color=color_palette[2], label='지배구조')

    # print(eco_per, society_per, governance_per)

    # 바 위에 값 표시하기
    e_label = [f"{eco_per:.1f}%" if eco_per > 0 else ""]
    s_label = [f"{society_per:.1f}%" if society_per > 0 else ""]
    g_label = [f"{governance_per:.1f}%" if governance_per > 0 else ""]

    # 범례 항목 생성을 위한 조건 추가
    legend_labels = []
    if eco_per > 0:
        legend_labels.append('환경')
    if society_per > 0:
        legend_labels.append('사회')
    if governance_per > 0:
        legend_labels.append('지배구조')

    plt.bar_label(bars1, label_type='center', labels=e_label)
    plt.bar_label(bars2, label_type='center', labels=s_label)
    plt.bar_label(bars3, label_type='center', labels=g_label)

    # 축 레이블, 범례 등 설정
    plt.title(f'{selected_brand}의 esg %')
    plt.yticks([])
    plt.annotate(f"총 {len(brand_df)} 건", (1, 1.05), xycoords='axes fraction', ha='right', fontsize=8, color='black')

    if len(legend_labels) > 0:
        plt.legend(legend_labels, bbox_to_anchor=(1.01, 1))
    st.pyplot(fig)

# ESG 지표별 영향력 비중 그래프
def influence_plt(brand_df):
    eco_df = brand_df[brand_df["esg_idx"] == "환경"][["esg_idx","영향력"]]
    society_df = brand_df[brand_df["esg_idx"] == "사회"][["esg_idx","영향력"]]
    governance_df = brand_df[brand_df["esg_idx"] == "지배구조"][["esg_idx","영향력"]]

    df_list = [eco_df, society_df, governance_df]

    for df in df_list:
        if not df.empty:
            vpos_count = (df["영향력"] == "매우 긍정적인 영향력").sum()
            pos_count = (df["영향력"] == "약간 긍정적인 영향력").sum()
            neg_count = (df["영향력"] == "약간 부정적인 영향력").sum()
            vneg_count = (df["영향력"] == "매우 부정적인 영향력").sum()

            total_count = vpos_count + pos_count + neg_count + vneg_count
            vpos_per = (vpos_count / total_count) * 100
            pos_per = (pos_count / total_count) * 100
            neg_per = (neg_count / total_count) * 100
            vneg_per = (vneg_count / total_count) * 100

            fig, ax = plt.subplots(figsize=(10, 2))
            categories=["ESG 지표별 영향력 %"]

            # 그래프 그리기
            bars1 = plt.barh(categories, vpos_per, color=color_palette[4], label='매우 긍정')
            bars2 = plt.barh(categories, pos_per, left=vpos_per, color=color_palette[3], label='약간 긍정')
            bars3 = plt.barh(categories, neg_per, left=vpos_per + pos_per, color=color_palette[2], label='약간 부정')
            bars4 = plt.barh(categories, vneg_per, left=vpos_per + pos_per + neg_per, color=color_palette[1], label='매우 부정')


            # 바 위에 값 표시하기
            vpos_label = [f"{vpos_per:.1f}%" if vpos_per > 0 else ""]
            pos_label = [f"{pos_per:.1f}%" if pos_per > 0 else ""]
            neg_label = [f"{neg_per:.1f}%" if neg_per > 0 else ""]
            vneg_label = [f"{vneg_per:.1f}%" if vneg_per > 0 else ""]

            # 범례 항목 생성을 위한 조건 추가
            legend_labels = []
            if vpos_per > 0:
                bars1[0].set_label('매우 긍정')
                legend_labels.append(bars1[0])
            if pos_per > 0:
                bars2[0].set_label('약간 긍정')
                legend_labels.append(bars2[0])
            if neg_per > 0:
                bars3[0].set_label('약간 부정')
                legend_labels.append(bars3[0])
            if vneg_per > 0:
                bars4[0].set_label('매우 부정')
                legend_labels.append(bars4[0])

            plt.bar_label(bars1, label_type='center', labels=vpos_label)
            plt.bar_label(bars2, label_type='center', labels=pos_label)
            plt.bar_label(bars3, label_type='center', labels=neg_label)
            plt.bar_label(bars4, label_type='center', labels=vneg_label)

            # 축 레이블, 범례 등 설정
            plt.title(f'{df["esg_idx"].values[0]} 영향력 %')
            plt.yticks([])
            plt.annotate(f"총 {total_count} 건", (1, 1.05), xycoords='axes fraction', ha='right', fontsize=10, color='black')

            plt.legend(handles=legend_labels, bbox_to_anchor=(1.01, 1))
            st.pyplot(fig)
        else:
            pass

# wordcloud 생성
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

# main news 시각화
def brand_main_news(brand_df):
    e_positive_df = brand_df[(brand_df["esg_idx"] == "환경") & (brand_df["영향력"] == "매우 긍정적인 영향력")].head(1)
    s_positive_df = brand_df[(brand_df["esg_idx"] == "사회") & (brand_df["영향력"] == "매우 긍정적인 영향력")].head(1)
    g_positive_df = brand_df[(brand_df["esg_idx"] == "지배구조") & (brand_df["영향력"] == "매우 긍정적인 영향력")].head(1)

    e_negative_df = brand_df[(brand_df["esg_idx"] == "환경") & (brand_df["영향력"] == "매우 부정적인 영향력")].head(1)
    s_negative_df = brand_df[(brand_df["esg_idx"] == "사회") & (brand_df["영향력"] == "매우 부정적인 영향력")].head(1)
    g_negative_df = brand_df[(brand_df["esg_idx"] == "지배구조") & (brand_df["영향력"] == "매우 부정적인 영향력")].head(1)

    e_brand_df = pd.concat([e_positive_df, e_negative_df], axis=0)
    s_brand_df = pd.concat([s_positive_df, s_negative_df], axis=0)
    g_brand_df = pd.concat([g_positive_df, g_negative_df], axis=0)

    esg_brand_df = pd.concat([e_brand_df, s_brand_df, g_brand_df], axis=0)

    return esg_brand_df

# ======================================================================================================================
# streamlit code
# ======================================================================================================================
with st.form("브랜드 뉴스 기사 조회"):
    st.title('브랜드 ESG뉴스 기사 조회')
    st.markdown("""
    원하는 브랜드를 검색하시면, 브랜드별로 ESG 구성 비중과 주요 키워드를 한 눈에 볼 수 있습니다.
    """)
    selected_brand = st.selectbox('브랜드 이름을 입력하세요:', brand, placeholder='Select...')
    submitted = st.form_submit_button("조회")

    if submitted:
        brand_df = filtered_brand_df(brand_name=selected_brand)
        brand_df.set_index("날짜", inplace=True)
        brand_df.sort_index(ascending=False, inplace=True)

        e_content_nouns_series = brand_df[brand_df["esg_idx"] == "환경"]["nouns"]
        s_content_nouns_series = brand_df[brand_df["esg_idx"] == "사회"]["nouns"]
        g_content_nouns_series = brand_df[brand_df["esg_idx"] == "지배구조"]["nouns"]

        esg_brand_df = brand_main_news(brand_df)

        if not e_content_nouns_series.empty:
            for content_nouns in e_content_nouns_series:
                brand_name_nouns = set(content_nouns).intersection(brand)
                for brand_name in brand_name_nouns:
                    c = Counter([x for x in content_nouns if x != brand_name])
                    e_word_count_dict[brand_name] += c

        if not s_content_nouns_series.empty:
            for content_nouns in s_content_nouns_series:
                brand_name_nouns = set(content_nouns).intersection(brand)
                for brand_name in brand_name_nouns:
                    c = Counter([x for x in content_nouns if x != brand_name])
                    s_word_count_dict[brand_name] += c

        if not g_content_nouns_series.empty:
            for content_nouns in g_content_nouns_series:
                brand_name_nouns = set(content_nouns).intersection(brand)
                for brand_name in brand_name_nouns:
                    c = Counter([x for x in content_nouns if x != brand_name])
                    g_word_count_dict[brand_name] += c

        if not brand_df.empty:
            if not e_content_nouns_series.empty:
                e_wordcloud_image = visualize_wordcloud(e_word_count_dict[selected_brand], "YlOrBr", e_mask_image)
            if not s_content_nouns_series.empty:
                s_wordcloud_image = visualize_wordcloud(s_word_count_dict[selected_brand], "YlOrBr", s_mask_image)
            if not g_content_nouns_series.empty:
                g_wordcloud_image = visualize_wordcloud(g_word_count_dict[selected_brand], "YlOrBr", g_mask_image)

            with st.spinner("데이터를 불러오는 중..."):
                st.write(f"{selected_brand} ESG 뉴스 top {len(esg_brand_df)}")
                st.dataframe(esg_brand_df[["제목", "esg_idx", "영향력", "url"]], column_config={"url": st.column_config.LinkColumn("Link")}, height=200, width=2000)
                plot_esg_spending(brand_df)
                influence_plt(brand_df)

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

        else:
            st.write("조회 되는 뉴스 기사가 없습니다")
    else:
        st.write("")

# selected_brand 전체기사 노출
on = st.toggle('전체 기사 목록')

brand_df = filtered_brand_df(brand_name=selected_brand)
brand_df.set_index("날짜", inplace=True)
brand_df.sort_index(ascending=False, inplace=True)

if on:
    st.dataframe(brand_df[["제목", "esg_idx", "영향력", "url"]], column_config={"url": st.column_config.LinkColumn("Link")}, height=200, width=2000)