import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===================== 页面全局配置 =====================
st.set_page_config(page_title="高端电商数据分析仪表盘", layout="wide")
st.title("📊 高端电商订单综合数据分析仪表盘")
st.caption("数据分析全流程：数据生成 → 数据清洗 → 多维统计 → 可视化挖掘 → 用户分层 → 异常检测")

# ===================== 1. 生成高质量模拟数据（pd.date_range） =====================
@st.cache_data
def create_data():
    np.random.seed(2026)
    # 全年连续日期生成（作业核心考点）
    date_range = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")

    region_list = ["华东", "华南", "华北", "西南", "华中", "东北", "西北"]
    product_list = ["手机", "笔记本电脑", "蓝牙耳机", "平板", "智能手表", "智能音箱", "游戏手柄"]
    channel_list = ["官网", "天猫", "京东", "抖音", "拼多多"]

    total_count = 1500
    df = pd.DataFrame({
        "订单号": [f"OD{str(i).zfill(6)}" for i in range(1, total_count+1)],
        "下单日期": np.random.choice(date_range, total_count),
        "销售地区": np.random.choice(region_list, total_count),
        "产品名称": np.random.choice(product_list, total_count),
        "销售渠道": np.random.choice(channel_list, total_count),
        "销售数量": np.random.randint(1, 30, total_count),
        "产品单价": np.random.randint(99, 8999, total_count),
        "用户ID": [f"USER{str(np.random.randint(1000,5000))}" for _ in range(total_count)]
    })

    # 人工制造脏数据（用于清洗实验）
    df.loc[10, "下单日期"] = np.nan
    df.loc[30, "销售地区"] = np.nan
    df.loc[60, "产品单价"] = np.nan
    df.loc[90, "销售数量"] = np.nan
    df.loc[180] = df.loc[50]
    df.loc[260] = df.loc[88]
    df.loc[350] = df.loc[120]

    return df

# ===================== 2. 完整数据清洗模块（dropna / drop_duplicates） =====================
def clean_data(df_raw):
    df = df_raw.copy()
    log_list = []
    log_list.append(f"原始数据总量：{len(df)} 行")

    # 日期标准化
    df["下单日期"] = pd.to_datetime(df["下单日期"], errors="coerce")

    # 缺失值统计
    miss_date = df["下单日期"].isna().sum()
    miss_area = df["销售地区"].isna().sum()
    miss_price = df["产品单价"].isna().sum()
    miss_num = df["销售数量"].isna().sum()
    log_list.append(f"检测缺失值：日期{miss_date}、地区{miss_area}、单价{miss_price}、销量{miss_num}")

    # 清除缺失值
    before = len(df)
    df = df.dropna(subset=["下单日期", "销售地区", "产品单价", "销售数量"])
    log_list.append(f"dropna 清理无效数据：移除 {before - len(df)} 行")

    # 清除重复订单
    before = len(df)
    df = df.drop_duplicates(subset=["订单号"], keep="first")
    log_list.append(f"drop_duplicates 清理重复订单：移除 {before - len(df)} 行")

    # 衍生字段
    df["订单总金额"] = df["销售数量"] * df["产品单价"]
    df["月份"] = df["下单日期"].dt.month
    df["季度"] = df["下单日期"].dt.quarter
    df["星期"] = df["下单日期"].dt.day_name()

    return df, log_list

# ===================== 3. 多维分组统计 groupby =====================
def group_statistics(df):
    # 地区统计
    area_group = df.groupby("销售地区").agg(
        订单总数=("订单号", "count"),
        总销量=("销售数量", "sum"),
        总销售额=("订单总金额", "sum"),
        平均客单价=("订单总金额", "mean")
    ).reset_index().sort_values("总销售额", ascending=False)

    # 产品统计
    product_group = df.groupby("产品名称").agg(
        订单数=("订单号", "count"),
        总销量=("销售数量", "sum"),
        总销售额=("订单总金额", "sum")
    ).reset_index().sort_values("总销售额", ascending=False)

    # 渠道统计
    channel_group = df.groupby("销售渠道").agg(
        订单数=("订单号", "count"),
        总销售额=("订单总金额", "sum")
    ).reset_index().sort_values("总销售额", ascending=False)

    # 月度统计
    month_group = df.groupby("月份").agg(
        订单量=("订单号", "count"),
        销售额=("订单总金额", "sum")
    ).reset_index()

    # 季度统计
    quarter_group = df.groupby("季度").agg(
        订单量=("订单号", "count"),
        销售额=("订单总金额", "sum")
    ).reset_index()

    # 地区+产品交叉统计
    cross_group = df.groupby(["销售地区", "产品名称"]).agg(
        销售额=("订单总金额", "sum"),
        销量=("销售数量", "sum")
    ).reset_index()

    return area_group, product_group, channel_group, month_group, quarter_group, cross_group

# ===================== 4. 透视表分析 pivot_table =====================
def pivot_analysis(df, threshold):
    high_sale_df = df[df["销售数量"] >= threshold]

    pivot_sale = pd.pivot_table(
        high_sale_df,
        index="销售地区",
        columns="产品名称",
        values="订单总金额",
        aggfunc="sum",
        fill_value=0
    )

    pivot_num = pd.pivot_table(
        high_sale_df,
        index="销售地区",
        columns="产品名称",
        values="销售数量",
        aggfunc="sum",
        fill_value=0
    )

    return high_sale_df, pivot_sale, pivot_num

# ===================== 5. 表合并 merge =====================
def merge_table(df, cross_data):
    merge_result = pd.merge(
        df,
        cross_data,
        on=["销售地区", "产品名称"],
        how="left",
        suffixes=("_明细", "_汇总")
    )
    return merge_result

# ===================== 6. RFM 用户价值分析 =====================
def rfm_analyse(df):
    latest = df["下单日期"].max()
    rfm = df.groupby("用户ID").agg(
        最近购买间隔=("下单日期", lambda x: (latest - x.max()).days),
        购买次数=("订单号", "count"),
        累计消费=("订单总金额", "sum")
    ).reset_index()

    rfm["用户等级"] = pd.cut(
        rfm["累计消费"],
        bins=[0, 3000, 10000, 30000, 999999],
        labels=["低价值用户", "中价值用户", "高价值用户", "超高价值用户"]
    )
    return rfm.sort_values("累计消费", ascending=False)

# ===================== 7. 异常数据检测 =====================
def error_detect(df):
    high_price = df[df["产品单价"] >= df["产品单价"].quantile(0.95)]
    high_sale = df[df["销售数量"] >= df["销售数量"].quantile(0.95)]
    top_order = df.sort_values("订单总金额", ascending=False).head(30)
    return high_price, high_sale, top_order

# ===================== 加载与处理数据 =====================
df_raw = create_data()
df_clean, clean_log = clean_data(df_raw)
area_df, pro_df, ch_df, mon_df, qt_df, cross_df = group_statistics(df_clean)

# 侧边栏筛选
st.sidebar.header("🔎 高级筛选面板")
area_sel = st.sidebar.multiselect("选择销售地区", df_clean["销售地区"].unique(), default=df_clean["销售地区"].unique())
pro_sel = st.sidebar.multiselect("选择产品", df_clean["产品名称"].unique(), default=df_clean["产品名称"].unique())
ch_sel = st.sidebar.multiselect("选择销售渠道", df_clean["销售渠道"].unique(), default=df_clean["销售渠道"].unique())
num_limit = st.sidebar.slider("高销量订单阈值", 3, 20, 8)

df_filter = df_clean[
    (df_clean["销售地区"].isin(area_sel)) &
    (df_clean["产品名称"].isin(pro_sel)) &
    (df_clean["销售渠道"].isin(ch_sel))
]

high_df, pivot_sale_tab, pivot_num_tab = pivot_analysis(df_filter, num_limit)
merge_all = merge_table(df_filter, cross_df)
rfm_df = rfm_analyse(df_filter)
err_price_df, err_num_df, top_order_df = error_detect(df_filter)

# ===================== 核心指标（表格版，彻底无省略号） =====================
st.subheader("📌 核心经营指标总览")
kpi_all = pd.DataFrame({
    "指标":["总订单量","总销量","总销售额","平均客单价","活跃用户数","日均销售额"],
    "数值":[
        f"{len(df_filter):,}",
        f"{df_filter['销售数量'].sum():,}",
        f"{df_filter['订单总金额'].sum():,.0f}",
        f"{df_filter['订单总金额'].mean():,.0f}",
        f"{df_filter['用户ID'].nunique():,}",
        f"{df_filter.groupby('下单日期')['订单总金额'].sum().mean():,.0f}"
    ]
})
st.dataframe(kpi_all.T, use_container_width=True, hide_index=True)
st.divider()

# ===================== 七大标签页 =====================
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "数据清洗与质量报告",
    "时间趋势分析",
    "地区销售分析",
    "产品渠道分析",
    "透视表深度分析",
    "用户RFM分层",
    "数据合并与异常检测"
])

# 标签1：数据清洗
with tab1:
    st.subheader("🧹 数据清洗全过程日志")
    for text in clean_log:
        st.info(text)
    st.subheader("清洗前后数据对比")
    comp = pd.DataFrame({"状态":["原始数据","清洗后数据"],"数据行数":[len(df_raw),len(df_clean)]})
    st.dataframe(comp, use_container_width=True)
    st.subheader("清洗后完整数据预览")
    st.dataframe(df_filter.head(200), use_container_width=True)

# 标签2：时间趋势
with tab2:
    st.subheader("📈 月度销售额 & 订单量趋势")
    fig1 = make_subplots(rows=2,cols=1)
    fig1.add_trace(go.Scatter(x=mon_df["月份"],y=mon_df["销售额"],name="月度销售额"),row=1,col=1)
    fig1.add_trace(go.Bar(x=mon_df["月份"],y=mon_df["订单量"],name="月度订单量"),row=2,col=1)
    st.plotly_chart(fig1,use_container_width=True)

    st.subheader("季度销售对比")
    fig2 = px.bar(qt_df,x="季度",y="销售额",color="季度")
    st.plotly_chart(fig2,use_container_width=True)

# 标签3：地区分析
with tab3:
    st.subheader("各地区销售额排名")
    st.dataframe(area_df,use_container_width=True)
    fig3 = px.bar(area_df,x="销售地区",y="总销售额",color="销售地区")
    st.plotly_chart(fig3,use_container_width=True)

# 标签4：产品渠道
with tab4:
    st.subheader("产品销售排行")
    st.dataframe(pro_df,use_container_width=True)
    fig4 = px.pie(pro_df,values="总销售额",names="产品名称")
    st.plotly_chart(fig4,use_container_width=True)

    st.subheader("渠道销售分布")
    fig5 = px.bar(ch_df,x="销售渠道",y="总销售额",color="销售渠道")
    st.plotly_chart(fig5,use_container_width=True)

# 标签5：透视表
with tab5:
    st.subheader("地区-产品销售额透视表")
    st.dataframe(pivot_sale_tab,use_container_width=True)
    st.subheader("地区-产品销量透视表")
    st.dataframe(pivot_num_tab,use_container_width=True)

# 标签6：用户RFM
with tab6:
    st.subheader("用户价值分层榜单")
    st.dataframe(rfm_df.head(50),use_container_width=True)
    user_class = rfm_df["用户等级"].value_counts().reset_index()
    fig6 = px.bar(user_class,x="用户等级",y="count",color="用户等级")
    st.plotly_chart(fig6,use_container_width=True)

# 标签7：合并+异常
with tab7:
    st.subheader("Merge 数据合并结果")
    st.dataframe(merge_all.head(100),use_container_width=True)

    st.subheader("高价异常订单")
    st.dataframe(err_price_df,use_container_width=True)
    st.subheader("高销量异常订单")
    st.dataframe(err_num_df,use_container_width=True)
    st.subheader("销售额TOP30订单")
    st.dataframe(top_order_df,use_container_width=True)

# 数据导出
st.divider()
st.subheader("📥 全套数据导出")
c1,c2,c3,c4 = st.columns(4)
with c1:st.download_button("清洗数据",df_filter.to_csv(index=False,encoding="utf-8-sig"),"清洗数据.csv")
with c2:st.download_button("地区统计",area_df.to_csv(index=False,encoding="utf-8-sig"),"地区统计.csv")
with c3:st.download_button("用户RFM",rfm_df.to_csv(index=False,encoding="utf-8-sig"),"用户RFM.csv")
with c4:st.download_button("透视表",pivot_sale_tab.to_csv(encoding="utf-8-sig"),"透视表数据.csv")


