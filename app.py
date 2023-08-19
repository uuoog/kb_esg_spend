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

# open ai key

os.environ["OPENAI_TOKEN"] = st.secrets["OPENAI_TOKEN"]
openai_token = os.environ.get("OPENAI_TOKEN")
if not openai_token:
    raise ValueError("no openai token!")
openai.api_key = openai_token


# plot font 설정
font_dirs = ['./fonts']
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)

fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'NanumGothicCoding'

# ======================================================================================================================
# ======================================================================================================================
# 변수 선언
color_palette = ["#ffa505", "#ffb805", "#ffc905", "#ffe505", "#fffb05"]
brand_df_col_dict = {
    '대분류': str,
    '중분류': str,
    '법인명': str,
    '브랜드명': str,
    'brno': str,
    'crno': str,
}

# corp_df_col_dict = {
#     '대분류': str,
#     'name': str,
#     'e_idx': str,
#     's_idx': str,
#     'g_idx': str,
#     'jurir_no': str,
#     'bizr_no': str,
# }

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

personality_dict = {
    "심쿵비비":"듬직하고 배려깊게 친구들을 잘 보듬어주는 다정다감하고 순둥순둥한 성격",
    "멜랑콜리":"항상 무심한 표정으로 뚱해보이지만 말 없이 친구들을 잘 챙기는 따뜻한 성격",
    "포스아거":"무모할 정도로 도전정신이 넘치고 허당미 뿜뿜한 성격",
    "롤로라무":"언제 어떤 일이 있어도 다 괜찮다고 생각하는 무한긍정 성격",
    "루나키키":"호불호가 분명하고 감정표현이 자유분방하며, 솔직하고 뒤 끝 없는 성격"
}

# ======================================================================================================================
# data 선언
spending_df = pd.read_csv("data/proecessed_spending_df.csv", encoding="utf-8")
brand_df = pd.read_csv("./data/brand_df.csv", dtype=brand_df_col_dict, encoding="utf-8-sig")
# corp_df = pd.read_csv("./data/corp_df.csv", dtype=corp_df_col_dict, encoding="utf-8")

with open("./data/brand_brno_dict.json", "r") as json_file:
    brand_brno_dict = json.load(json_file)

with open("./data/brand_ctg_dict.json", "r") as json_file:
    brand_ctg_dict = json.load(json_file)
# ======================================================================================================================

st.title("KB ESG")
# image = Image.open("data/image/kbimg.jpg")
# st.image(image)

st.write("Write Something")
# @kbkookminbank



# ======================================================================================================================
# 함수 선언
# ======================================================================================================================

# esg 소비액, 이용 브랜드명, 업종 칼럼 추가
@st.cache_resource
def add_col(spending_df):
    grade_dict = {
        "S": 1,
        "A+": 0.85,
        "A": 0.71,
        "B+": 0.57,
        "B": 0.42,
        "C": 0.28,
        "D": 0.14,
    }

    brand_list = []

    brand_df["이용 브랜드"] = ""
    brand_df["카테고리"] = ""

    spending_df["환경(E) 소비"] = 0
    spending_df["사회(S) 소비"] = 0
    spending_df["지배구조(S) 소비"] = 0

    for i in tqdm(range(len(spending_df))):

        for j in range(len(brand_df)):

            if brand_df.iloc[j]["브랜드명"].replace(" ", "") in spending_df.iloc[i]["이용하신곳"].replace(" ", ""):
                brand_name = brand_df.iloc[j]["브랜드명"]
                spending_df.loc[i, "이용 브랜드"] = brand_name

                ctg = brand_ctg_dict[brand_name]
                spending_df.loc[i, "카테고리"] = ctg

                # br_no = brand_brno_dict[brand_name]
                #
                # if not corp_df[corp_df["bizr_no"] == br_no]["name"].empty:
                #     brand_list.append((i, brand_name, corp_df[corp_df["bizr_no"] == br_no]["name"].item()))

                    # 등급별 가중치로 소비액 계산
                    # spending_money = spending_df.iloc[i]["국내이용금액 (원)"]

                    # # 기업별 등급 가져오기
                    # if not corp_df[corp_df["bizr_no"] == br_no]["e_idx"].empty:
                    #     e_idx = corp_df.loc[corp_df["bizr_no"] == br_no, "e_idx"].item()
                    #     spending_df.loc[i, "환경(E) 소비"] = spending_money * grade_dict[e_idx]
                    #
                    # if not corp_df[corp_df["bizr_no"] == br_no]["s_idx"].empty:
                    #     s_idx = corp_df.loc[corp_df["bizr_no"] == br_no, "s_idx"].item()
                    #     spending_df.loc[i, "사회(S) 소비"] = spending_money * grade_dict[s_idx]
                    #
                    # if not corp_df[corp_df["bizr_no"] == br_no]["g_idx"].empty:
                    #     g_idx = corp_df.loc[corp_df["bizr_no"] == br_no, "g_idx"].item()
                    #     spending_df.loc[i, "지배구조(S) 소비"] = spending_money * grade_dict[g_idx]

                    # break

    spending_df[["환경(E) 소비", "사회(S) 소비", "지배구조(S) 소비"]] = spending_df[["환경(E) 소비", "사회(S) 소비", "지배구조(S) 소비"]].fillna(0)

    return spending_df


# esg 소비액 계산

def cal_esg_spending(spending_df):
    # 소비액 계산
    esg_spending_dict = {
        "환경(E) 소비": 0,
        "사회(S) 소비": 0,
        "지배구조(S) 소비": 0
    }

    spending_total = spending_df["국내이용금액 (원)"].sum()

    esg_spending_dict["환경(E) 소비"] = spending_df["환경(E) 소비"].sum()
    esg_spending_dict["사회(S) 소비"] = spending_df["사회(S) 소비"].sum()
    esg_spending_dict["지배구조(S) 소비"] = spending_df["지배구조(S) 소비"].sum()

    e_spending_per = round(esg_spending_dict["환경(E) 소비"] * 100 / spending_total, 2)
    s_spending_per = round(esg_spending_dict["사회(S) 소비"] * 100 / spending_total, 2)
    g_spending_per = round(esg_spending_dict["지배구조(S) 소비"] * 100 / spending_total, 2)

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
    bars3 = plt.barh(categories, g_spending_per, left=e_spending_per + s_spending_per, color=color_palette[2], label='지배구조 지수')
    plt.barh(categories, 100 - (e_spending_per + s_spending_per + g_spending_per),
             left=e_spending_per + s_spending_per + g_spending_per, color=color_palette[0], label='전체 소비')

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
@st.cache_resource
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

@st.cache_resource
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
# streamlit code
# ======================================================================================================================

with st.form("고객 정보 조회"):
    st.title('고객 정보 조회')
    customer_list = ["차국민", "라국민", "허리브", "정국민", "현국민", "강리브"]
    selected_name = st.selectbox('고객 이름', customer_list)
    submitted = st.form_submit_button("조회")

    if submitted:
        choosed_df = filtered_spending_df(selected_name)

        if not choosed_df.empty:
            # spending_df 칼럼 추가
            choosed_df = add_col(choosed_df)
            choosed_df_show = choosed_df.drop(["년", "월", "일", "국내이용금액 (원)"], axis=1)
            # dataframe 출력
            st.write(choosed_df_show)
            # st.write(choosed_df)


            # ESG 지표별 소비액 계산
            spending_total, esg_spending_dict, e_spending_per, s_spending_per, g_spending_per = cal_esg_spending(
                choosed_df)

            # ESG 지표중 최고 지표 기준의 결제 df 생성
            max_key, max_esg_spending_df = make_max_esg_spending_df(choosed_df)


            formatted_total = "{:,}".format(choosed_df["국내이용금액 (원)"].sum())

            # 이용 고객 ESG 소비 비중 그래프
            plot_esg_spending()
            # 이용 고객 최고 ESG 소비 비중 TOP3 업종 그래프
            plot_max_esg_ctg()

            if max_esg_spending_df.iloc[0].name in eng_cat_dict:
                eng_place = eng_cat_dict[max_esg_spending_df.iloc[0].name]

            place = eng_place
            img, character = get_openai_image(place)
            # st.write(get_openai_image(place))
            st.write(img)

            spending_summary = f"""{selected_name}님의 7월 ESG 소비 내역입니다.

            (그래프)

            총 소비액: {spending_total}원
            환경(E) 소비액: {esg_spending_dict["환경(E) 소비"]}원 (전체 소비 대비 {e_spending_per}%)
            사회(S) 소비액: {esg_spending_dict["사회(S) 소비"]}원 (전체 소비 대비 {s_spending_per}%)
            지배구조(G) 소비액: {esg_spending_dict["지배구조(S) 소비"]}원 (전체 소비 대비: {g_spending_per}%)

            (AI 이미지)
            {selected_name}님은 {max_key} 지킴이!
            -이하 프롬프트-
            """

            st.write(spending_summary)
            # openai_image = get_openai_image(place)

            # prompt
            prompt = generate_prompt(character, selected_name, place, max_key)
            st.write(request_chat_completion(character, prompt))



        else:
            st.write(f"{selected_name}에 대한 정보가 없습니다.")

