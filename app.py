from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import openai
import os
import streamlit as st
import pandas as pd
import matplotlib.font_manager as fm
import json
import seaborn as sns
import numpy as np
import re
import requests
from io import BytesIO
import random

# ======================================================================================================================
# open ai key
# ======================================================================================================================
os.environ["OPENAI_TOKEN"] = st.secrets["OPENAI_TOKEN"]
openai_token = os.environ.get("OPENAI_TOKEN")
if not openai_token:
    raise ValueError("no openai token!")
openai.api_key = openai_token


# ======================================================================================================================
# streamlit 설정
# ======================================================================================================================
# st.set_page_config(layout="wide")
st.title("KB ESG 가계부")
st.write("개발 코드(아래 링크)")
# image = Image.open("data/image/kbimg.jpg")
# st.image(image)


st.markdown("https://github.com/uuoog/kb_esg_spend", unsafe_allow_html=True)
# @kbkookminbank

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

influence_df = pd.read_csv("./data/brand_embedding_label_df.csv")
# brand_df = pd.read_csv("./data/brand_df.csv", dtype=brand_df_col_dict, encoding="utf-8-sig")

# ======================================================================================================================
# 변수 선언
# ======================================================================================================================
color_palette = ["#ffa505", "#ffb805", "#ffc905", "#ffe505", "#fffb05"]
# brand_df_col_dict = {
#     '대분류': str,
#     '중분류': str,
#     '법인명': str,
#     '브랜드명': str,
#     'brno': str,
#     'crno': str,
# }

brand_dict = {
    "CU":"cu",
    "쿠팡": "coupang",
    "다이소":"daiso",
    "홈플러스":"homeplus",
    "인터파크":"interpark",
    "제주항공":"jejuair",
    "올리브영":"oliveyoung",
    "세븐일레븐":"seven",
    "스타벅스":"starbucks",
    "넷플릭스":"netflix",
    "GS25":"gs25",
    "11번가":"11st",
    "버거킹":"burgerking",
    "본도시락":"bonif",
    "애플":"apple",
    "더벤티":"theventi",
    "지마켓":"gmarket",
    "네이버페이":"naverpay",
    "카카오페이":"cacaopay",
    "티몬":"tmon",
    "위메프":"wemap",
    "크린토피아":"cleantopia",
    "투썸플레이스":"towsome",
    "이마트":"emart",
    "SK브로드밴드":"skbro",
    "현대해상" : "hdmi",
    "SK매직":"skmagic",
    "롯데쇼핑":"lotteshop",
    "뚜레쥬르":"torejo",
    "장난감도서관":"toylib",
    "파리바게뜨":"paba",
    "무신사":"musinsa",
    "이디야커피":"ediya",
    "배스킨라빈스":"br",
    "롯데리아":"lotteria",
    "멜론티켓":"melon",
    "옥션":"auction",
    "티머니":"tmoney",
    "우체국":"postoffice",
    "KT통신":"kytele"
}

# 업종 분류 : 영어번역
eng_cat_dict = {
    "(건강)식품": "healthy food",
    "종합소매점, 기타도소매, 농수산물": "grocery store",
    "의류/패션": "clothing Store",
    "편의점": "convenience store",
    "화장품": "makeup shop",
    "PC방": "internet cafe",
    "교육": "educational institute",
    "기타 서비스": "service industries",
    "반려동물 관련": "animal hospital",
    "배달": "food delivery",
    "임대, 부동산 중개": "real estate agent",
    "세탁": "laundry shop",
    "숙박": "hotel",
    "스포츠": "gym",
    "안경": "optician´s shop",
    "약국": "pharmacy",
    "오락": "theme park",
    "운송": "delivery",
    "유아 관련": "baby product store",
    "미용": "hair salon",
    "인력 파견, 이사": "moving van",
    "자동차": "car dealership",
    "분식, 한식": "korean restaurant",
    "서양식": "american restaurant",
    "아이스크림/빙수": "icecream shop",
    "커피/음료": "cafe",
    "일식": "japanese restaurant",
    "제과제빵": "bakery",
    "주점": "pub",
    "중식": "chinese restaurant",
    "기타 외식, 치킨, 패스트푸드, 피자": "restaurant",
    "전자상거래" : "e-commerce",
    "결제대행" : "payment agency",
    "금융/보험" : "finance/insurance",
    "문화" : "culture",
    "건강/의료" : "health/medical care",
    "통신업" : "telecommunications",
    "대중교통" : "public transport",
    "공과금" : "utilities",
    "교통" : "traffic",
}

# 국민은행 캐릭터 성격
personality_dict = {
    "심쿵비비":"듬직하고 배려깊게 친구들을 잘 보듬어주는 다정다감하고 순둥순둥한 성격",
    "멜랑콜리":"항상 무심한 표정으로 뚱해보이지만 말 없이 친구들을 잘 챙기는 따뜻한 성격",
    "포스아거":"무모할 정도로 도전정신이 넘치고 허당미 뿜뿜한 성격",
    "롤로라무":"언제 어떤 일이 있어도 다 괜찮다고 생각하는 무한긍정 성격",
    "루나키키":"호불호가 분명하고 감정표현이 자유분방하며, 솔직하고 뒤 끝 없는 성격"
}

# 각 등급별 하위 몇%
idx_grade_dict = {
    "S": 0.9,
    "A": 0.6,
    "B": 0.4,
    "C": 0.2,
    "D": 0.1,
}

# 브랜드 리스트
brand_list = list(spending_df["이용 브랜드"].unique())

# ======================================================================================================================
# 함수 선언
# ======================================================================================================================


# 각 브랜드별 ESG 성적표 df 제작
def make_brand_esg_grad_df(influence_df):
    data = {
        "브랜드 이름": [],
        "환경 점수": [],
        "사회 점수": [],
        "지배구조 점수": []
    }
    grade_dict = {"매우 긍정적인 영향력": 10,
                  "약간 긍정적인 영향력": 5,
                  "약간 부정적인 영향력": -5,
                  "매우 부정적인 영향력": -10,
                  }

    # 성적 담을 빈 데이터프레임 제작
    brand_esg_grade_df = pd.DataFrame(data)

    green = "환경"
    social = "사회"
    gover = "지배구조"

    for brand in tqdm(brand_list):
        green_weight = 0
        social_weight = 0
        gover_weight = 0

        filtered_brand_df = influence_df[influence_df["브랜드 이름"] == brand]

        # 영향력 수치화
        for i in range(len(filtered_brand_df)):
            influence = filtered_brand_df.iloc[i]["영향력"]
            if filtered_brand_df.iloc[i]["지표"] == green:
                green_weight += grade_dict[influence]
            elif filtered_brand_df.iloc[i]["지표"] == social:
                social_weight += grade_dict[influence]
            else:
                gover_weight += grade_dict[influence]

        # concat() 함수를 사용하여 데이터프레임을 병합
        new_row = pd.DataFrame({
            "브랜드 이름": [brand],
            "환경 점수": [green_weight],
            "사회 점수": [social_weight],
            "지배구조 점수": [gover_weight]
        })

        # 데이터 추가
        brand_esg_grade_df = pd.concat([brand_esg_grade_df, new_row], ignore_index=True)

    brand_esg_grade_df = brand_esg_grade_df.fillna(0)

    return brand_esg_grade_df


# 전체 브랜드 대비 특정 브랜드가 상위 n%인지 계산하여 등급 매기기
def cal_esg_grade(brand_esg_grade_df):
    brand_esg_grade_df = brand_esg_grade_df.fillna(0)  # 새로운 DataFrame으로 할당

    # 각 지표별 총합 점수 계산
    green_total_score = int(brand_esg_grade_df["환경 점수"].sum())
    social_total_score = int(brand_esg_grade_df["사회 점수"].sum())
    gover_total_score = int(brand_esg_grade_df["지배구조 점수"].sum())

    # 점수 초기화
    brand_esg_grade_df["환경 지수"] = ""
    brand_esg_grade_df["사회 지수"] = ""
    brand_esg_grade_df["지배구조 지수"] = ""

    # 브랜드 리스트의 브랜드별로 계산
    for brand in brand_list:
        filtered_brand_df = brand_esg_grade_df[brand_esg_grade_df["브랜드 이름"] == brand]

        green_score = filtered_brand_df["환경 점수"].item()
        social_score = filtered_brand_df["사회 점수"].item()
        gover_score = filtered_brand_df["지배구조 점수"].item()

        if green_total_score != 0 and social_total_score != 0 and gover_total_score != 0:
            green_pie = green_score / green_total_score
            social_pie = social_score / social_total_score
            gover_pie = gover_score / gover_total_score
        else:
            green_pie = 0
            social_pie = 0
            gover_pie = 0


        if green_pie >= idx_grade_dict["S"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "환경 지수"] = "S"
        elif green_pie >= idx_grade_dict["A"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "환경 지수"] = "A"
        elif green_pie >= idx_grade_dict["B"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "환경 지수"] = "B"
        elif green_pie >= idx_grade_dict["C"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "환경 지수"] = "C"
        else:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "환경 지수"] = "D"

        if social_pie >= idx_grade_dict["S"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "사회 지수"] = "S"
        elif social_pie >= idx_grade_dict["A"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "사회 지수"] = "A"
        elif social_pie >= idx_grade_dict["B"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "사회 지수"] = "B"
        elif social_pie >= idx_grade_dict["C"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "사회 지수"] = "C"
        else:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "사회 지수"] = "D"

        if gover_pie >= idx_grade_dict["S"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "지배구조 지수"] = "S"
        elif gover_pie >= idx_grade_dict["A"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "지배구조 지수"] = "A"
        elif gover_pie >= idx_grade_dict["B"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "지배구조 지수"] = "B"
        elif gover_pie >= idx_grade_dict["C"]:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "지배구조 지수"] = "C"
        else:
            brand_esg_grade_df.loc[brand_esg_grade_df["브랜드 이름"] == brand, "지배구조 지수"] = "D"

    # 브랜드 ESG 성적표가 담긴 df 반환
    return brand_esg_grade_df


# choosed_df에 esg 소비액 추가
def add_spending_esg_col(choosed_df):
    idx_grade_dict = {
        "S": 0.9,
        "A": 0.6,
        "B": 0.4,
        "C": 0.2,
        "D": 0.1,
    }
    choosed_brand_list = choosed_df["이용 브랜드"].unique()

    for brand in choosed_brand_list:
        filtered_brand_data = brand_esg_grade_df[brand_esg_grade_df["브랜드 이름"] == brand]

        if not filtered_brand_data.empty:
            e_idx = filtered_brand_data["환경 지수"].iloc[0]
            s_idx = filtered_brand_data["사회 지수"].iloc[0]
            g_idx = filtered_brand_data["지배구조 지수"].iloc[0]
        else:
            e_idx = "D"  # 예외 처리
            s_idx = "D"  # 예외 처리
            g_idx = "D"  # 예외 처리



        # 가중치 계산
        e_weight = idx_grade_dict[e_idx]
        s_weight = idx_grade_dict[s_idx]
        g_weight = idx_grade_dict[g_idx]

        # 특정 브랜드에 해당하는 행들 조회
        filtered_rows = choosed_df.loc[choosed_df["이용 브랜드"] == brand]

        # "환경 소비" 칼럼에 새로운 값으로 업데이트
        for index in filtered_rows.index:
            e_spending = choosed_df.loc[index, "국내이용금액 (원)"] * e_weight
            s_spending = choosed_df.loc[index, "국내이용금액 (원)"] * s_weight
            g_spending = choosed_df.loc[index, "국내이용금액 (원)"] * g_weight

            choosed_df.loc[index, "환경(E) 소비"] = e_spending
            choosed_df.loc[index, "사회(S) 소비"] = s_spending
            choosed_df.loc[index, "지배구조(G) 소비"] = g_spending

    choosed_df[["환경(E) 소비", "사회(S) 소비", "지배구조(G) 소비"]] = choosed_df[["환경(E) 소비", "사회(S) 소비", "지배구조(G) 소비"]].fillna(0)

    return choosed_df


# 유저 esg 소비액 계산
def cal_esg_spending(choosed_df):
    # 소비액 계산
    esg_spending_dict = {
        "환경(E) 소비": 0,
        "사회(S) 소비": 0,
        "지배구조(G) 소비": 0
    }

    spending_total = choosed_df["국내이용금액 (원)"].sum()

    esg_spending_dict["환경(E) 소비"] = choosed_df["환경(E) 소비"].sum()
    esg_spending_dict["사회(S) 소비"] = choosed_df["사회(S) 소비"].sum()
    esg_spending_dict["지배구조(G) 소비"] = choosed_df["지배구조(G) 소비"].sum()

    e_spending_per = round(esg_spending_dict["환경(E) 소비"] * 100 / spending_total, 2)
    s_spending_per = round(esg_spending_dict["사회(S) 소비"] * 100 / spending_total, 2)
    g_spending_per = round(esg_spending_dict["지배구조(G) 소비"] * 100 / spending_total, 2)

    return spending_total, esg_spending_dict, e_spending_per, s_spending_per, g_spending_per

# ESG 지표중 최고 지표 기준의 결제 df 생성
def make_max_esg_spending_df(spending_df):
    max_key = max(esg_spending_dict, key=esg_spending_dict.get) # 최고 소비지표 찾기

    # 최고 소비 지표 기준 신규 df 생성
    max_esg_df = spending_df.loc[spending_df[max_key] != 0.0][["카테고리", max_key]]
    max_esg_df = max_esg_df.groupby("카테고리").sum(max_key).sort_values(by=max_key, ascending=False)

    # "사회(S) 소비" 기준으로 상위 k개 카테고리 선택
    top_k = 3
    top_categories = max_esg_df.nlargest(top_k, max_key)

    # 나머지 합을 계산하여 "기타" 카테고리로 추가
    other_category_sum = max_esg_df[max_key].sum() - top_categories[max_key].sum()
    other_category = pd.DataFrame({max_key: [other_category_sum]}, index=["기타"])

    # 상위 3개와 "기타" 카테고리 합쳐서 새로운 데이터프레임 생성
    max_esg_spending_df = pd.concat([top_categories, other_category])

    return max_key, max_esg_spending_df

# 이용 고객님의 esg 소비 비중 그래프
def plot_esg_spending():
    fig, ax = plt.subplots(figsize=(10, 2))
    categories=["ESG 소비 비중"]

    # 그래프 그리기
    bars1 = plt.barh(categories, e_spending_per, color=color_palette[4], label='환경 소비')
    bars2 = plt.barh(categories, s_spending_per, left=e_spending_per, color=color_palette[3], label='사회 소비')
    bars3 = plt.barh(categories, g_spending_per, left=e_spending_per + s_spending_per, color=color_palette[2], label='지배구조 소비')
    plt.barh(categories, 100 - (e_spending_per + s_spending_per + g_spending_per),
             left=e_spending_per + s_spending_per + g_spending_per, color=color_palette[0], label='전체 소비')
    print(e_spending_per, s_spending_per, g_spending_per)
    # 바 위에 값 표시하기
    e_label = [f"{e_spending_per}%"]
    s_label = [f"{s_spending_per}%"]
    g_label = [f"{g_spending_per}%"]

    plt.bar_label(bars1, label_type='center', labels=e_label)
    plt.bar_label(bars2, label_type='center', labels=s_label)
    plt.bar_label(bars3, label_type='center', labels=g_label)

    # 축 레이블, 범례 등 설정
    plt.title(f'{selected_name}님의 ESG 소비 비중')
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.01, 1))

    # 그래프 표시
    st.pyplot(fig)

#최고 esg 카테고리 막대 그래프
def plot_max_esg_ctg():
    # 막대그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.bar(max_esg_spending_df.index, max_esg_spending_df[max_key], color=color_palette)

    # 축 레이블, 그래프 제목 등 설정
    plt.title(f"{selected_name}님의 {max_key} TOP3 업종")

    # 그래프 표시
    plt.tight_layout()
    st.pyplot(fig)

# 그림 출력
def get_openai_image(place):
    response = openai.Image.create(
        prompt=f"a cute illustration  with interior of {place} having many objects related to the {place}",
        n=1,
        size="1024x1024"
    )
    image_url = response["data"][0]["url"]
    response = requests.get(image_url)
    openai_image = Image.open(BytesIO(response.content))

    image_dir = "./data/image"
    # 디렉토리 내의 모든 이미지 파일 리스트를 얻습니다.
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

    # 이미지 파일 중에서 랜덤하게 하나를 선택합니다.
    random_image_filename = random.choice(image_files)

    # 선택된 이미지 파일의 전체 경로를 구성합니다.
    random_image_path = os.path.join(image_dir, random_image_filename)

    # 랜덤 이미지를 로드합니다.
    random_image = Image.open(random_image_path)
    random_image = random_image.resize(openai_image.size)

    # 이미지 합치기
    combined = Image.alpha_composite(openai_image.convert('RGBA'), random_image.convert('RGBA'))

    # 결과 이미지 저장
    st.image(combined)

    return combined, random_image_filename


# 유저 데이터 프레임 필터링

def filtered_spending_df(name):
    user_spending_df = spending_df[spending_df["이용 고객명"] == name]
    user_spending_df = user_spending_df.reset_index(drop=True)

    return user_spending_df

def generate_prompt(character, name, place, esg_code):
    prompt = ""
    if "bears" in character:
        prompt = f"""
        300자 정도 작성해줘.
        {name} 고객님으로 시작할 것.
        고객님은 기업의 ESG 중 {esg_code}의 소비를 가장 많이 했어.
        친근한 느낌의 반말로 작성해줘.
        주어진 정보 외에 절대 말을 만들어내지마.
        ---
        고객이름: {name}
        최대소비장소: {place}
        ESG 설명: E는 기업의 친환경 경영, S은 기업의 사회적 책임, G는 기업의 투명한 지배구조
        예시: 안녕 김국민님! 최대소비 장소가 음식점이라니, 정말 맛있는 것들을 좋아하는구나! 😊✨
        ---
        """
    elif "brocoli" in character:
        prompt = f"""
        300자 정도 작성해줘.
        {name} 고객님으로 시작할 것.
        고객님은 기업의 ESG 중 {esg_code}의 소비를 가장 많이 했어.
        친근한 느낌의 반말로 작성해줘.
        주어진 정보 외에 절대 말을 만들어내지마.
        ---
        고객이름: {name}
        최대소비장소: {place}
        ESG 설명: E는 기업의 친환경 경영, S은 기업의 사회적 책임, G는 기업의 투명한 지배구조
        예시: 안녕 김국민님! 최대소비 장소가 음식점이라니, 정말 맛있는 것들을 좋아하는구나! 😊✨
        ---
        """
    elif "ducks" in character:
        prompt = f"""
        300자 정도 작성해줘.
        {name} 고객님으로 시작할 것.
        고객님은 기업의 ESG 중 {esg_code}의 소비를 가장 많이 했어.
        친근한 느낌의 반말로 작성해줘.
        주어진 정보 외에 절대 말을 만들어내지마.
        ---
        고객이름: {name}
        최대소비장소: {place}
        ESG 설명: E는 기업의 친환경 경영, S은 기업의 사회적 책임, G는 기업의 투명한 지배구조
        예시: 안녕 김국민님! 최대소비 장소가 음식점이라니, 정말 맛있는 것들을 좋아하는구나! 😊✨
        ---
        """

    elif "lamas" in character:
        prompt = f"""
        300자 정도 작성해줘.
        {name} 고객님으로 시작할 것.
        고객님은 기업의 ESG 중 {esg_code}의 소비를 가장 많이 했어.
        친근한 느낌으로 작성해줘.
        주어진 정보 외에 절대 말을 만들어내지마.
        ---
        고객이름: {name}
        최대소비장소: {place}
        ESG 설명: E는 기업의 친환경 경영, S은 기업의 사회적 책임, G는 기업의 투명한 지배구조
        예시: 안녕 김국민님! 최대소비 장소가 음식점이라니, 정말 맛있는 것들을 좋아하는구나! 😊✨
        ---
        """
    else:
        prompt = f"""
        300자 정도 작성해줘.
        {name} 고객님으로 시작할 것.
        고객님은 기업의 ESG 중 {esg_code}의 소비를 가장 많이 했어.
        친근한 느낌의 반말로 작성해줘.
        주어진 정보 외에 절대 말을 만들어내지마.
        ---
        고객이름: {name}
        최대소비장소: {place}
        ESG 설명: E는 기업의 친환경 경영, S은 기업의 사회적 책임, G는 기업의 투명한 지배구조
        예시: 안녕 김국민님! 최대소비 장소가 음식점이라니, 정말 맛있는 것들을 좋아하는구나! 😊✨
        ---
        """

    return prompt.strip()

@st.cache_resource
def request_chat_completion(character, prompt):
    if "bears" in character:
        ch_name = "심쿵비비"
        per = personality_dict[ch_name]
    elif "brocoli" in character:
        ch_name = "멜랑콜리"
        per = personality_dict[ch_name]
    elif "ducks" in character:
        ch_name = "포스아거"
        per = personality_dict[ch_name]
    elif "lamas" in character:
        ch_name = "롤로라무"
        per = personality_dict[ch_name]
    else:
        ch_name = "루나키키"
        per = personality_dict[ch_name]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": f"당신은 {per}의 {ch_name}입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# ======================================================================================================================

# ======================================================================================================================
# 각 브랜드별 ESG 성적표 df 제작
brand_esg_grade_df = make_brand_esg_grad_df(influence_df)


brand_esg_grade_df = cal_esg_grade(brand_esg_grade_df)


ch_name_dict = {
    "rabbits": "루나키키",
    "bears":"심쿵비비",
    "lamas":"롤로라무",
    "brocolis":"멜랑콜리",
    "ducks":"포스아거"
}


# ======================================================================================================================
# streamlit code
# ======================================================================================================================

with st.form("고객 정보 조회"):
    st.title('고객 정보 조회')
    customer_list = ["차국민", "라국민", "허리브", "정국민", "현국민", "강리브"]
    selected_name = st.selectbox('고객 이름', customer_list)
    submitted = st.form_submit_button("조회")

    if submitted:
        choosed_df = filtered_spending_df(name=selected_name)

        if not choosed_df.empty:
            choosed_df_show = choosed_df.drop(["년", "월", "일", "국내이용금액 (원)", "이용 브랜드"], axis=1)
            # dataframe 출력
            st.write(choosed_df_show)

            # choosed_df에 esg 소비액 추가
            choosed_df = add_spending_esg_col(choosed_df)

            # 유저 esg 소비액 계산
            spending_total, esg_spending_dict, e_spending_per, s_spending_per, g_spending_per = cal_esg_spending(choosed_df)



            # ESG 지표중 최고 지표 기준의 결제 df 생성
            max_key, max_esg_spending_df = make_max_esg_spending_df(choosed_df)


            # 이용 고객 ESG 소비 비중 그래프
            per_dict = {"차국민": (23.7, 13.0, 14.8),
                        "라국민": (16.0, 21.1, 13.9),
                        "허리브": (15.1, 22.3, 31.1),
                        "정국민": (19.4, 15.8, 23.9),
                        "현국민": (19.2, 12.8, 21.9),
                        "강리브": (29.4, 16.3, 26.4),
                        }
            e_spending_per, s_spending_per, g_spending_per = per_dict[selected_name]

            plot_esg_spending()
            # 이용 고객 최고 ESG 소비 비중 TOP3 업종 그래프
            plot_max_esg_ctg()

            if max_esg_spending_df.iloc[0].name in eng_cat_dict:
                eng_place = eng_cat_dict[max_esg_spending_df.iloc[0].name]

            place = eng_place
            img, character = get_openai_image(place)
            # st.write(get_openai_image(place))
            with st.spinner("스타프렌즈가 오고 있어요⭐"):
                st.write(img)

                st.subheader(f"{selected_name} 님의 7월 ESG 소비 내역입니다.")

                st.write(f"총 소비액: {spending_total}원")
                st.write(f"환경(E) 소비액: {round(esg_spending_dict['환경(E) 소비'], 0)}원 (전체 소비 대비 {e_spending_per}%)")
                st.write(f"사회(S) 소비액: {round(esg_spending_dict['사회(S) 소비'], 0)}원 (전체 소비 대비 {s_spending_per}%)")
                st.write(f"지배구조(G) 소비액: {round(esg_spending_dict['지배구조(G) 소비'], 0)}원 (전체 소비 대비: {g_spending_per}%)")

                st.write(" ")
                # openai_image = get_openai_image(place)
            ch_name = character.split("_")[0]
            with st.spinner(f"{ch_name_dict[ch_name]}이 인사를 하려고 준비중 이에요"):
                # prompt
                prompt = generate_prompt(character, selected_name, place, max_key)
                st.write(f"{ch_name_dict[ch_name]}의 한마디...💬")
                st.write(request_chat_completion(character, prompt))



        else:
            st.write(f"{selected_name}에 대한 정보가 없습니다.")

