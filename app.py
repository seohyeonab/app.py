
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

MIN_OBSERVATION_DAYS = 300
LAST_YEAR = 2025

# 학습 / 테스트 기간
TRAIN_50_START = 1956
TRAIN_100_START = 1906
TRAIN_END = 2005

TEST_START = 2006
TEST_END = 2025


# ==================================================
# 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜 처리
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 처리
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.dropna(
        subset=["연평균기온"]
    )

    return annual


annual = load_data()


# ==================================================
# 회귀 모델 함수
# ==================================================
def make_model(train_data):

    X = train_data[["연도"]]
    y = train_data["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def evaluate_model(model, test_data):

    X_test = test_data[["연도"]]
    y_test = test_data["연평균기온"]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, mse, r2, predictions


# ==================================================
# 전체 데이터 모델
# ==================================================
whole_model = make_model(annual)

whole_predictions = whole_model.predict(
    annual[["연도"]]
)

whole_mae = mean_absolute_error(
    annual["연평균기온"],
    whole_predictions
)

whole_mse = mean_squared_error(
    annual["연평균기온"],
    whole_predictions
)

whole_r2 = r2_score(
    annual["연평균기온"],
    whole_predictions
)

whole_slope = whole_model.coef_[0]


# ==================================================
# 학습 데이터
# ==================================================

# 최근 50년: 1956~2005
train_50 = annual[
    (annual["연도"] >= TRAIN_50_START) &
    (annual["연도"] <= TRAIN_END)
].copy()

# 최근 100년: 1906~2005
train_100 = annual[
    (annual["연도"] >= TRAIN_100_START) &
    (annual["연도"] <= TRAIN_END)
].copy()


# ==================================================
# 공통 테스트 데이터: 2006~2025
# ==================================================
test = annual[
    (annual["연도"] >= TEST_START) &
    (annual["연도"] <= TEST_END)
].copy()


# ==================================================
# 50년 모델
# ==================================================
model_50 = make_model(train_50)

mae_50, mse_50, r2_50, pred_50 = evaluate_model(
    model_50,
    test
)

slope_50 = model_50.coef_[0]


# ==================================================
# 100년 모델
# ==================================================
model_100 = make_model(train_100)

mae_100, mse_100, r2_100, pred_100 = evaluate_model(
    model_100,
    test
)

slope_100 = model_100.coef_[0]


# ==================================================
# 화면
# ==================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온을 선형회귀 모델로 학습하고, "
    "학습에 사용하지 않은 최근 20년 데이터를 이용해 "
    "모델의 예측 성능을 평가합니다."
)


# ==================================================
# 데이터 구성
# ==================================================
st.subheader("📚 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{train_50['연도'].min()}~{train_50['연도'].max()}"
    )
    st.caption(
        f"{len(train_50)}개 연도"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        f"{train_100['연도'].min()}~{train_100['연도'].max()}"
    )
    st.caption(
        f"{len(train_100)}개 연도"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"{test['연도'].min()}~{test['연도'].max()}"
    )
    st.caption(
        f"{len(test)}개 연도"
    )

st.info(
    "두 모델 모두 2006~2025년의 기온을 학습하지 않고, "
    "이 기간을 공통 테스트 데이터로 사용합니다."
)


# ==================================================
# 전체 데이터 평가
# ==================================================
st.subheader("📊 전체 데이터로 학습했을 때")

st.write(
    "전체 유효 연도 데이터를 모두 학습한 회귀 모델을 "
    "같은 데이터에 적용한 결과입니다."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "기울기",
        f"{whole_slope * 100:+.3f} ℃/100년"
    )

with col2:
    st.metric(
        "MAE",
        f"{whole_mae:.3f} ℃"
    )

with col3:
    st.metric(
        "MSE",
        f"{whole_mse:.3f}"
    )

with col4:
    st.metric(
        "R²",
        f"{whole_r2:.3f}"
    )

st.caption(
    "전체 데이터 평가는 학습 데이터와 평가 데이터가 동일하므로 "
    "실제 미래 예측 성능을 평가하는 값과는 다릅니다."
)


# ==================================================
# 50년 vs 100년 비교
# ==================================================
st.subheader("🔎 최근 50년 학습 vs 최근 100년 학습")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🟦 최근 50년 모델")

    st.metric(
        "회귀선 기울기",
        f"{slope_50 * 100:+.3f} ℃/100년"
    )

    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )

    st.caption(
        "학습: 1956~2005년 → 테스트: 2006~2025년"
    )


with col2:

    st.markdown("### 🟧 최근 100년 모델")

    st.metric(
        "회귀선 기울기",
        f"{slope_100 * 100:+.3f} ℃/100년"
    )

    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )

    st.caption(
        "학습: 1906~2005년 → 테스트: 2006~2025년"
    )


# ==================================================
# 성능 비교 해석
# ==================================================
st.subheader("💡 두 모델의 성능 비교")

if mae_50 < mae_100:
    better_mae = "최근 50년 모델"
else:
    better_mae = "최근 100년 모델"

if mse_50 < mse_100:
    better_mse = "최근 50년 모델"
else:
    better_mse = "최근 100년 모델"

if r2_50 > r2_100:
    better_r2 = "최근 50년 모델"
else:
    better_r2 = "최근 100년 모델"

st.write(
    f"- **MAE가 더 작은 모델:** {better_mae}"
)

st.write(
    f"- **MSE가 더 작은 모델:** {better_mse}"
)

st.write(
    f"- **R²가 더 큰 모델:** {better_r2}"
)

slope_difference = (
    (slope_50 - slope_100) * 100
)

if slope_difference > 0:
    st.write(
        f"- 최근 50년 모델의 기울기가 최근 100년 모델보다 "
        f"100년당 **{slope_difference:.3f}℃ 더 큽니다.**"
    )
else:
    st.write(
        f"- 최근 100년 모델의 기울기가 최근 50년 모델보다 "
        f"100년당 **{abs(slope_difference):.3f}℃ 더 큽니다.**"
    )


# ==================================================
# 테스트 데이터 실제값 vs 예측값
# ==================================================
st.subheader("📈 테스트 기간 실제 기온과 예측값")

fig = go.Figure()

# 실제 기온
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

# 50년 모델 예측
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="최근 50년 학습 모델",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

# 100년 모델 예측
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="최근 100년 학습 모델",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    legend_title="구분"
)

fig.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 테스트 데이터 예측 결과 표
# ==================================================
st.subheader("📋 테스트 데이터 예측 결과")

result = test[
    ["연도", "연평균기온"]
].copy()

result["50년 모델 예측"] = pred_50
result["100년 모델 예측"] = pred_100

result["50년 모델 오차"] = (
    result["50년 모델 예측"]
    - result["연평균기온"]
)

result["100년 모델 오차"] = (
    result["100년 모델 예측"]
    - result["연평균기온"]
)

st.dataframe(
    result.round(2),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 기울기 설명
# ==================================================
st.subheader("📐 회귀선 기울기 해석")

st.write(
    f"**최근 50년 모델:** "
    f"100년에 {slope_50 * 100:+.3f}℃"
)

st.write(
    f"**최근 100년 모델:** "
    f"100년에 {slope_100 * 100:+.3f}℃"
)

st.caption(
    "기울기는 연도 1년당 예상 기온 변화량을 100배하여 "
    "'100년에 몇 ℃ 변하는가'로 표시했습니다."
)


# ==================================================
# 참고
# ==================================================
st.caption(
    "※ 연평균기온 관측일수가 300일 미만인 연도는 분석에서 제외했습니다. "
    "또한 1906년 데이터가 제공되지 않는 경우 실제 학습에는 "
    "데이터가 존재하는 가장 이른 연도부터 사용됩니다."
)
