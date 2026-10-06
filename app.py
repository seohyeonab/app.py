import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


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
TRAIN_END = 2005
TEST_START = 2006
TEST_END = 2025

# 고차 회귀 계산용 기준 연도
# 실제 연도 대신 (연도 - 2005)를 사용해서
# 큰 숫자를 그대로 거듭제곱하는 문제를 줄임
BASE_YEAR = 2005


# ==================================================
# 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온
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
        df["연도"] <= TEST_END
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

    # 회귀 계산용 변환 연도
    annual["계산연도"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# ==================================================
# 훈련 / 테스트 분리
# ==================================================

# 훈련 데이터: 2005년 이전
train = annual[
    annual["연도"] <= TRAIN_END
].copy()

# 테스트 데이터: 2006~2025
test = annual[
    (annual["연도"] >= TEST_START) &
    (annual["연도"] <= TEST_END)
].copy()


# ==================================================
# 다항 회귀 함수
# ==================================================
def polynomial_fit(train_data, degree):

    x = train_data["계산연도"].to_numpy()
    y = train_data["연평균기온"].to_numpy()

    # 다항식 계수 계산
    coefficients = np.polyfit(
        x,
        y,
        degree
    )

    return coefficients


def predict(coefficients, years):

    x = np.asarray(years) - BASE_YEAR

    return np.polyval(
        coefficients,
        x
    )


# ==================================================
# MAE 계산
# ==================================================
def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(actual - predicted)
    )


# ==================================================
# 1차 / 3차 / 9차 모델 학습
# ==================================================
models = {
    "1차": polynomial_fit(train, 1),
    "3차": polynomial_fit(train, 3),
    "9차": polynomial_fit(train, 9)
}


# ==================================================
# 테스트 데이터 평가
# ==================================================
actual = test["연평균기온"].to_numpy()

results = []

for degree_name, coefficients in models.items():

    predictions = predict(
        coefficients,
        test["연도"].to_numpy()
    )

    mae = calculate_mae(
        actual,
        predictions
    )

    prediction_2050 = predict(
        coefficients,
        [2050]
    )[0]

    results.append({
        "모델": degree_name,
        "테스트 평균 오차 (MAE)": mae,
        "2050년 예상 기온 (℃)": prediction_2050
    })


results_df = pd.DataFrame(results)


# ==================================================
# 제목
# ==================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온을 이용해 1차, 3차, 9차 다항 회귀 모델을 학습하고 "
    "학습에 사용하지 않은 2006~2025년 데이터로 예측 성능을 평가합니다."
)


# ==================================================
# 데이터 개수
# ==================================================
st.subheader("📚 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "훈련 데이터",
        f"{len(train)}개 연도"
    )

    st.caption(
        f"{train['연도'].min()}~{train['연도'].max()}년"
    )


with col2:

    st.metric(
        "테스트 데이터",
        f"{len(test)}개 연도"
    )

    st.caption(
        f"{test['연도'].min()}~{test['연도'].max()}년"
    )


st.info(
    "2005년까지의 데이터만 모델을 학습하는 데 사용하고, "
    "2006~2025년 데이터는 학습 과정에서 전혀 사용하지 않습니다. "
    "따라서 테스트 성능은 학습에 사용하지 않은 데이터로 평가합니다."
)


# ==================================================
# 결과 표
# ==================================================
st.subheader("📊 모델별 테스트 성능과 2050년 예측")

display_df = results_df.copy()

display_df["테스트 평균 오차 (MAE)"] = (
    display_df["테스트 평균 오차 (MAE)"]
    .map(lambda x: f"{x:.3f} ℃")
)

display_df["2050년 예상 기온 (℃)"] = (
    display_df["2050년 예상 기온 (℃)"]
    .map(lambda x: f"{x:.2f} ℃")
)

st.table(display_df)


st.caption(
    "MAE는 테스트 기간의 실제 연평균기온과 예측값의 차이를 "
    "절댓값으로 계산한 평균입니다. 작을수록 테스트 예측이 정확합니다."
)


# ==================================================
# 테스트 데이터 실제값 + 예측 곡선
# ==================================================
st.subheader("📈 테스트 기간 실제 기온과 모델 예측")

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


# 테스트 기간의 모델 예측
test_years = test["연도"].to_numpy()

for degree_name, coefficients in models.items():

    predictions = predict(
        coefficients,
        test_years
    )

    fig.add_trace(
        go.Scatter(
            x=test_years,
            y=predictions,
            mode="lines",
            name=f"{degree_name} 회귀",
            hovertemplate=(
                "<b>%{x}년</b><br>"
                f"{degree_name} 예측: "
                "%{y:.2f} ℃"
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
# 전체 기간에서 회귀 곡선이 어떻게 이어지는지
# ==================================================
st.subheader("📉 훈련 기간부터 2050년까지 회귀 곡선")

curve_years = np.arange(
    train["연도"].min(),
    2051
)

fig2 = go.Figure()


# 실제 데이터
fig2.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["연평균기온"],
        mode="markers",
        name="훈련 데이터",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 테스트 실제 데이터
fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트 데이터",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 1차 / 3차 / 9차 곡선
for degree_name, coefficients in models.items():

    curve_predictions = predict(
        coefficients,
        curve_years
    )

    fig2.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_predictions,
            mode="lines",
            name=f"{degree_name} 회귀곡선",
            hovertemplate=(
                "<b>%{x}년</b><br>"
                f"{degree_name}: "
                "%{y:.2f} ℃"
                "<extra></extra>"
            )
        )
    )


fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="구분"
)

fig2.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ==================================================
# 모델별 해석
# ==================================================
st.subheader("💡 모델 비교")

best_model = results_df.loc[
    results_df["테스트 평균 오차 (MAE)"].idxmin(),
    "모델"
]

best_mae = results_df[
    results_df["모델"] == best_model
]["테스트 평균 오차 (MAE)"].iloc[0]

st.write(
    f"테스트 데이터(2006~2025년)에서 MAE가 가장 작은 모델은 "
    f"**{best_model} 회귀**이며, 평균적으로 약 "
    f"**{best_mae:.2f}℃** 차이가 났습니다."
)

st.write(
    "1차 회귀는 전체적인 상승·하락 추세를 단순하게 표현하고, "
    "3차와 9차 회귀는 훈련 기간의 굴곡까지 더 자세히 따라갈 수 있습니다."
)

st.warning(
    "특히 9차처럼 차수가 높은 모델은 훈련 데이터의 작은 변동까지 "
    "따라가면서 테스트 기간이나 2050년처럼 학습 범위를 벗어난 구간에서 "
    "예측이 크게 흔들릴 수 있습니다."
)


# ==================================================
# 계산 방법 설명
# ==================================================
st.subheader("🧮 계산 방법")

st.write(
    "고차 회귀의 수치적인 불안정을 줄이기 위해 실제 연도 자체를 "
    "거듭제곱하지 않고 **계산연도 = 연도 − 2005**로 변환했습니다."
)

st.write(
    "예를 들어 2005년은 0, 2006년은 1, 2050년은 45가 되어 "
    "9차식에서도 비교적 작은 숫자로 계산됩니다."
)

st.caption(
    "※ 관측일수가 300일 미만인 연도는 연평균기온 계산 및 분석에서 제외했습니다."
)
