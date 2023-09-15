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

# 이용 고객님의 esg 소비 비중 그래프
def plot_esg_spending(brand_df):

    eco_count = (brand_df["지표"] == "환경").sum()
    society_count = (brand_df["지표"] == "사회").sum()
    governance_count = (brand_df["지표"] == "지배구조").sum()

    total_count = brand_df["지표"].count()
    eco_per = (eco_count / total_count) * 100
    society_per = (society_count / total_count) * 100
    governance_per = (governance_count / total_count) * 100

    fig, ax = plt.subplots(figsize=(6, 2))
    categories = ["ESG 소비 비중"]

    # 그래프 그리기
    bars1 = plt.barh(categories, eco_per, color=color_palette[4], label='환경 소비')
    bars2 = plt.barh(categories, society_per, left=eco_per, color=color_palette[3], label='사회 소비')
    bars3 = plt.barh(categories, governance_per, left=eco_per + society_per, color=color_palette[2], label='지배구조 소비')

    # print(eco_per, society_per, governance_per)

    # 바 위에 값 표시하기
    e_label = [f"{eco_per:.1f}%" if eco_per > 0 else ""]
    s_label = [f"{society_per:.1f}%" if society_per > 0 else ""]
    g_label = [f"{governance_per:.1f}%" if governance_per > 0 else ""]

    # 범례 항목 생성을 위한 조건 추가
    legend_labels = []
    if eco_per > 0:
        legend_labels.append('환경 소비')
    if society_per > 0:
        legend_labels.append('사회 소비')
    if governance_per > 0:
        legend_labels.append('지배구조 소비')

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
    eco_df = brand_df[brand_df["지표"] == "환경"][["지표","영향력"]]
    society_df = brand_df[brand_df["지표"] == "사회"][["지표","영향력"]]
    governance_df = brand_df[brand_df["지표"] == "지배구조"][["지표","영향력"]]

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
                legend_labels.append('매우 긍정')
            if pos_per > 0:
                legend_labels.append('약간 긍정')
            if neg_per > 0:
                legend_labels.append('약간 부정')
            if vneg_per > 0:
                legend_labels.append('매우 부정')

            plt.bar_label(bars1, label_type='center', labels=vpos_label)
            plt.bar_label(bars2, label_type='center', labels=pos_label)
            plt.bar_label(bars3, label_type='center', labels=neg_label)
            plt.bar_label(bars4, label_type='center', labels=vneg_label)

            # 축 레이블, 범례 등 설정
            plt.title(f'{df["지표"].values[0]} 영향력 %')
            plt.yticks([])
            plt.annotate(f"총 {total_count} 건", (1, 1.05), xycoords='axes fraction', ha='right', fontsize=10, color='black')

            if len(legend_labels) > 0:
                plt.legend(legend_labels, bbox_to_anchor=(1.01, 1))
            st.pyplot(fig)
        else:
            pass

with st.form("브랜드 뉴스 기사 조회"):
    st.title('브랜드 뉴스 기사 조회')
    # brand_list = st.session_state.brand_list
    selected_brand = st.text_input("브랜드 이름을 입력하세요:")
    submitted2 = st.form_submit_button("조회")

    if submitted2:
        brand_df = filtered_brand_df(brand_name=selected_brand)
        brand_df.set_index("날짜", inplace=True)
        brand_df.sort_index(ascending=False, inplace=True)

        if not brand_df.empty:
            with st.spinner("데이터를 불러오는 중..."):
                st.dataframe(brand_df[["제목", "지표", "영향력"]], height=200)
                plot_esg_spending(brand_df)
                influence_plt(brand_df)
        if brand_df.empty:
            st.write("조회 되는 뉴스 기사가 없습니다")
