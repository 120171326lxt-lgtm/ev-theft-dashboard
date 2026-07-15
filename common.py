"""
公共组件:顶部品牌栏、指标卡样式、通用 CSS。
所有页面统一 import 使用，保证三张页面视觉风格一致（红色 ST 品牌 + 卡片式布局）。
"""
import streamlit as st

BRAND_RED = "#e5342b"


def inject_base_css():
    st.markdown(
        f"""
        <style>
            /* 收紧默认留白，贴近截图里的紧凑仪表盘风格 */
            .block-container {{
                padding-top: 3.5rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }}
            /* 顶部品牌栏 */
            .topbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #ffffff;
                border-bottom: 1px solid #eaeaea;
                padding: 10px 18px;
                margin: 0 -1rem 1.2rem -1rem;
                position: relative;
                z-index: 1;
            }}
            .topbar-left {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 15px;
                color: #1f2329;
            }}
            .topbar-logo {{
                background: {BRAND_RED};
                color: white;
                font-weight: 700;
                font-size: 13px;
                padding: 4px 8px;
                border-radius: 4px;
            }}
            .topbar-crumb-sep {{
                color: #c8cad0;
            }}
            .topbar-right {{
                font-size: 13px;
                color: #8a8f98;
            }}
            /* 指标卡 */
            .metric-card {{
                background: #ffffff;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 16px 18px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: 700;
                color: #1f2329;
            }}
            .metric-value.red {{ color: {BRAND_RED}; }}
            .metric-value.green {{ color: #2e9e5b; }}
            .metric-label {{
                font-size: 13px;
                color: #6b7076;
                margin-top: 4px;
            }}
            .metric-sub {{
                font-size: 12px;
                margin-top: 4px;
            }}
            .metric-sub.up {{ color: #2e9e5b; }}
            .metric-sub.down {{ color: {BRAND_RED}; }}
            /* 徽标标签 */
            .badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
            }}
            .badge-red {{ background: #fdecea; color: {BRAND_RED}; }}
            .badge-orange {{ background: #fff4e5; color: #c76b00; }}
            .badge-yellow {{ background: #fff9e0; color: #9c7a00; }}
            .badge-green {{ background: #e8f7ee; color: #2e9e5b; }}
            .badge-grey {{ background: #f0f1f3; color: #6b7076; }}
            table.section-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            table.section-table th {{
                background: {BRAND_RED};
                color: white;
                padding: 8px 10px;
                text-align: left;
            }}
            table.section-table td {{
                padding: 8px 10px;
                border-bottom: 1px solid #f0f0f0;
            }}
            table.section-table tr:hover td {{ background: #fafafa; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def topbar(crumb: str, right_text: str = ""):
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <span class="topbar-logo">ST</span>
                <span>电动自行车盗窃时空预测—预警平台</span>
                <span class="topbar-crumb-sep">/</span>
                <span>{crumb}</span>
            </div>
            <div class="topbar-right">{right_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(value: str, label: str, sub: str = "", value_color: str = "", sub_direction: str = ""):
    """value_color: '' | 'red' | 'green'   sub_direction: '' | 'up' | 'down'"""
    value_cls = f"metric-value {value_color}".strip()
    sub_cls = f"metric-sub {sub_direction}".strip()
    sub_html = f'<div class="{sub_cls}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="{value_cls}">{value}</div>
            <div class="metric-label">{label}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    mapping = {
        "预警激增": "badge-red",
        "高风险": "badge-orange",
        "中高风险": "badge-yellow",
        "低风险": "badge-green",
        "严重不足": "badge-red",
        "不足": "badge-orange",
        "临界": "badge-yellow",
        "达标": "badge-green",
        "P1": "badge-red",
        "P2": "badge-orange",
        "P3": "badge-grey",
    }
    cls = mapping.get(level, "badge-grey")
    return f'<span class="badge {cls}">{level}</span>'
