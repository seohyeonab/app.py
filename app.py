import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_CLASS_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜와 평균기온 정리
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 수업 기준 기간: 2025년까지
    df = df[df["연도"] <= LAST_CLASS_YEAR]

    # 연도별 통계
    # 관측일은 평균기온 값이 실제로 존재하는 날을 기준으로 계산
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    annual = annual[annual["관측일수"] >= MIN_OBSERVATION_DAYS].copy()

    # 회귀분석에 사용할 결측치 제거
    annual = annual.dropna(subset=["연평균기온"])

    # 회귀분석 독립 변수:
    # 1908년 -> 0, 1909년 -> 1, ...
    annual["1908년부터_지난_연수"] = annual["연도"] - BASE_YEAR

    return annual


annual = load_data()


# --------------------------------------------------
# 회귀 분석
# --------------------------------------------------
x = annual["1908년부터_지난_연수"].to_numpy()
y = annual["연평균기온"].to_numpy()

# y = slope * x + intercept
slope, intercept = np.polyfit(x, y, 1)

annual["회귀예측기온"] = slope * x + intercept

# 연도와 연평균기온의 피어슨 상관계수
correlation = annual["연도"].corr(annual["연평균기온"])

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연도별 평균기온을 이용해 기온 변화 추세를 살펴보고, "
    "선형 회귀로 선택한 연도의 예상 평균기온을 계산합니다."
)

# 데이터 정보
st.subheader("📌 회귀 직선에 사용한 데이터")

col1, col2, col3, col4 = st.columns(4)

col1.metric("사용한 연도 수", f"{year_count}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.3f}")

st.caption(
    f"2025년 이후 데이터와 연간 평균기온 관측일이 "
    f"{MIN_OBSERVATION_DAYS}일 미만인 해는 제외했습니다."
)


# --------------------------------------------------
# 산점도 + 회귀 직선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

# 회귀 직선
line_years = np.arange(start_year, end_year + 1)
line_x = line_years - BASE_YEAR
line_temperature = slope * line_x + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperature,
        mode="lines",
        name="선형 회귀 직선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선 기온: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="데이터",
)

# 가로축에는 '지난 연수'가 아니라 실제 연도를 표시
fig.update_xaxes(
    tickformat="d",
    range=[start_year - 2, end_year + 2],
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("🧮 회귀 분석")

st.write(
    f"상관계수: **{correlation:.4f}**"
)

st.write(
    f"회귀식: **예상 기온 = "
    f"{slope:.5f} × (연도 - {BASE_YEAR}) "
    f"{intercept:+.3f}**"
)


# --------------------------------------------------
# 연도 선택 및 예측
# --------------------------------------------------
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - BASE_YEAR
predicted_temperature = slope * selected_x + intercept

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-top: 15px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 60px;
            font-weight: bold;
            margin-top: 5px;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 관측 기간 밖이면 설명 표시
if selected_year < start_year or selected_year > end_year:
    st.warning(
        f"{selected_year}년은 회귀 직선에 사용한 관측 기간 "
        f"({start_year}~{end_year}년) 밖의 값이므로 "
        "회귀 직선을 연장한 추정값입니다."
    )

st.caption(
    "이 값은 과거 서울 기온의 선형 추세를 단순히 연장한 값이며, "
    "실제 미래 기후를 정밀하게 예측하는 기후모형의 결과는 아닙니다."
)
