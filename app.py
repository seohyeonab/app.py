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
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= LAST_CLASS_YEAR]

    # 연도별 평균기온 + 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.dropna(subset=["연평균기온"])

    return annual


annual = load_data()


# --------------------------------------------------
# 회귀 함수
# --------------------------------------------------
def calculate_regression(data):
    x = data["연도"].to_numpy() - BASE_YEAR
    y = data["연평균기온"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    # ℃/년 → 100년에 몇 ℃
    increase_per_100_years = slope * 100

    correlation = data["연도"].corr(data["연평균기온"])

    return slope, intercept, increase_per_100_years, correlation


# --------------------------------------------------
# 전체 기간 회귀
# --------------------------------------------------
slope_all, intercept_all, increase_all, correlation_all = (
    calculate_regression(annual)
)

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


# --------------------------------------------------
# 최근 20년 회귀
# --------------------------------------------------
recent_start = end_year - 19

recent_20 = annual[
    annual["연도"] >= recent_start
].copy()

slope_recent, intercept_recent, increase_recent, correlation_recent = (
    calculate_regression(recent_20)
)

recent_year_count = len(recent_20)
recent_actual_start = int(recent_20["연도"].min())
recent_actual_end = int(recent_20["연도"].max())


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용해 장기간의 기온 변화 추세를 살펴보고, "
    "회귀 직선을 이용해 선택한 연도의 예상 평균기온을 계산합니다."
)


# --------------------------------------------------
# 전체 기간 / 최근 20년 비교
# --------------------------------------------------
st.subheader("📊 기온 상승 속도 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")

    st.metric(
        "100년에 기온이 얼마나 변하는가?",
        f"{increase_all:+.2f} ℃",
    )

    st.caption(
        f"{start_year}~{end_year}년 · "
        f"{year_count}개 연도"
    )

    st.write(
        f"상관계수: **{correlation_all:.3f}**"
    )


with col2:
    st.markdown("### 최근 20년")

    st.metric(
        "100년에 기온이 얼마나 변하는가?",
        f"{increase_recent:+.2f} ℃",
    )

    st.caption(
        f"{recent_actual_start}~{recent_actual_end}년 · "
        f"{recent_year_count}개 연도"
    )

    st.write(
        f"상관계수: **{correlation_recent:.3f}**"
    )


st.info(
    "기울기는 ℃/년으로 계산한 뒤 100을 곱해 "
    "'100년에 몇 ℃ 변하는가'로 표시했습니다."
)


# --------------------------------------------------
# 산점도 + 전체 회귀선 + 최근 20년 회귀선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 전체 연도 산점도
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


# 전체 기간 회귀선
line_years_all = np.arange(
    start_year,
    end_year + 1
)

line_x_all = line_years_all - BASE_YEAR
line_y_all = slope_all * line_x_all + intercept_all

fig.add_trace(
    go.Scatter(
        x=line_years_all,
        y=line_y_all,
        mode="lines",
        name="전체 기간 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "전체 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


# 최근 20년 회귀선
line_years_recent = np.arange(
    recent_actual_start,
    recent_actual_end + 1
)

line_x_recent = line_years_recent - BASE_YEAR
line_y_recent = (
    slope_recent * line_x_recent
    + intercept_recent
)

fig.add_trace(
    go.Scatter(
        x=line_years_recent,
        y=line_y_recent,
        mode="lines",
        name="최근 20년 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="구분",
)

fig.update_xaxes(
    tickformat="d",
    range=[start_year - 2, end_year + 2],
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("🧮 회귀 분석 결과")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")
    st.write(
        f"**100년당 변화량: {increase_all:+.3f} ℃**"
    )
    st.write(
        f"상관계수: **{correlation_all:.4f}**"
    )

with col2:
    st.markdown("### 최근 20년")
    st.write(
        f"**100년당 변화량: {increase_recent:+.3f} ℃**"
    )
    st.write(
        f"상관계수: **{correlation_recent:.4f}**"
    )


# --------------------------------------------------
# 연도 선택 및 예측
# --------------------------------------------------
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요."

