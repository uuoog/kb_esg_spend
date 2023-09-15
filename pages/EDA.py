import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

# ======================================================================================================================
# streamlit 설정
# ======================================================================================================================
# st.title("DATA EDA")

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
influence_df = pd.read_csv("./data/brand_embedding_label_df_re.csv")

# ======================================================================================================================
# 변수 선언
# ======================================================================================================================
color_palette = ["#ffa505", "#ffb805", "#ffc905", "#ffe505", "#fffb05"]

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
    brand_df = influence_df[influence_df["브랜드 이름"] == brand_name]
    brand_df = brand_df.reset_index(drop=True)
    return brand_df


# 고객별 카테고리 bar_plot
def category_plt(choosed_df):
    category_counts = choosed_df["카테고리"].value_counts()
    threshold = 5
    category_counts_less_than_threshold = category_counts[category_counts < threshold]
    choosed_df["카테고리"] = choosed_df["카테고리"].apply(lambda x: "기타" if x in category_counts_less_than_threshold else x)
    category_counts = choosed_df["카테고리"].value_counts()
    # 그래프 크기 설정
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = plt.bar(category_counts.index, category_counts.values, color=color_palette)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=8)
        plt.title(f"{selected_name} 고객님의 카테고리 별 빈도수")
        plt.xlabel("카테고리")
        # 차트 아래에 주석 추가
        plt.annotate("5건 미만 카테고리 '기타'로 통합", (1, -0.12), xycoords='axes fraction', ha='right', fontsize=8, color='gray')
        plt.yticks([])
    st.pyplot(fig)

    result_df = choosed_df.groupby("카테고리")["국내이용금액 (원)"].sum().reset_index()
    top_5_brands = result_df.nlargest(5, "국내이용금액 (원)")
    # bar 차트 그리기
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = plt.bar(top_5_brands["카테고리"], top_5_brands["국내이용금액 (원)"], color=color_palette)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, format_with_commas(yval), ha='center', va='bottom',
                 fontsize=8)
        plt.title("카테고리 별 국내이용금액")
        plt.xlabel("카테고리")
        plt.ylabel("국내이용금액 (원)")
        plt.yticks([])

    # y축 레이블 포맷 지정
    formatter = FuncFormatter(format_with_commas)
    plt.gca().yaxis.set_major_formatter(formatter)
    # plt.xticks(rotation=45)  # x축 레이블을 45도 회전하여 표시
    st.pyplot(fig)


# 고객별 브랜드 top5 bar_plot
def brand_plt(choosed_df):
    top5_brand = choosed_df["이용 브랜드"].value_counts().head(5)

    # 그래프 설정
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = plt.bar(top5_brand.index, top5_brand.values, color=color_palette)
    for bar in bars:
        yval = bar.get_height()
        # plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=8)
        plt.text(bar.get_x() + bar.get_width() / 2, yval, format_with_commas(yval), ha='center', va='bottom', fontsize=8)
        plt.title(f"{selected_name} 고객님의 이용 브랜드 top5 빈도수")
        plt.xlabel("이용 브랜드")
        plt.ylabel("빈도수")
        plt.yticks([])
    # 그래프 표시
    st.pyplot(fig)

    result_df = choosed_df.groupby("이용 브랜드")["국내이용금액 (원)"].sum().reset_index()
    top_5_brands = result_df.nlargest(5, "국내이용금액 (원)")
    # 그래프 설정
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = plt.bar(top_5_brands["이용 브랜드"], top_5_brands["국내이용금액 (원)"], color=color_palette)
    for bar in bars:
        yval = bar.get_height()
        # plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=8)
        plt.text(bar.get_x() + bar.get_width() / 2, yval, format_with_commas(yval), ha='center', va='bottom', fontsize=8)
        plt.title("이용 브랜드 별 상위 5개의 국내이용금액")
        plt.xlabel("이용 브랜드")
        plt.ylabel("국내이용금액 (원)")
        plt.yticks([])
    # 그래프 표시
    st.pyplot(fig)


# ======================================================================================================================
# streamlit code
# ======================================================================================================================

with st.form("결제 내역 EDA"):
    st.title('결제 내역 EDA')
    customer_list = ["차국민", "라국민", "허리브", "정국민", "현국민", "강리브"]
    selected_name = st.selectbox('고객 이름', customer_list)
    submitted = st.form_submit_button("조회")

    if submitted:
        choosed_df = filtered_spending_df(name=selected_name)
        st.session_state.brand_list = list(choosed_df["이용 브랜드"].unique())

        # 요약
        st.markdown(f"{selected_name}의 요약 정보")

        if not choosed_df.empty:
            category_plt(choosed_df)
            brand_plt(choosed_df)

