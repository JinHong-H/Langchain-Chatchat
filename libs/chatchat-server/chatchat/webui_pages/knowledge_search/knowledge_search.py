import streamlit as st
import pandas as pd
from typing import List, Dict
from chatchat.server.utils import api_address
from chatchat.webui_pages.utils import *

def knowledge_search_page(api: ApiRequest):
    """知识库检索页面"""
    st.header("🔍 知识库检索")
    
    # 获取所有知识库列表
    kbs = api.list_knowledge_bases()
    kb_names = [kb["kb_name"] for kb in kbs] if isinstance(kbs, list) else []
    
    if not kb_names:
        st.warning("暂无可用知识库，请先创建知识库")
        return
    
    # 检索参数设置 - 分成两行
    selected_kb = st.selectbox("选择知识库", kb_names, key="search_kb")
    
    cols = st.columns(2)
    with cols[0]:
        top_k = st.number_input("返回数量", min_value=1, max_value=100, value=10, key="search_top_k")
    with cols[1]:
        score_threshold = st.slider("分数阈值", min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                                 help="分数越小相关度越高，0-2之间，建议0.5左右")
    
    # 搜索查询输入
    query = st.text_area("检索查询", placeholder="请输入要检索的内容...", height=100, key="search_query")
    
    # 执行搜索按钮
    if st.button("🔍 开始检索", type="primary", use_container_width=True):
        if not query.strip():
            st.error("请输入检索查询内容")
            return
        
        with st.spinner("正在检索..."):
            # 调用知识库搜索API
            result = api.search_kb_docs(
                knowledge_base_name=selected_kb,
                query=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if isinstance(result, list):
                st.session_state.search_results = result
                st.session_state.show_results = True
                st.session_state.selected_doc = None  # 清空之前选择的行
            else:
                st.error(f"检索失败: {result.get('msg', '未知错误')}")
    
    # 显示结果
    if st.session_state.get("show_results") and st.session_state.get("search_results"):
        display_search_results(st.session_state.search_results)
    

def display_search_results(docs: List[Dict]):
    """显示检索结果"""
    st.subheader("📊 检索结果")
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    
    # 准备显示数据
    results_data = []
    for i, doc in enumerate(docs):
        content = doc.get("page_content", "")[:200] + "..." if len(doc.get("page_content", "")) > 200 else doc.get("page_content", "")
        
        results_data.append({
            "序号": i + 1,
            "内容预览": content,
            "源文件": doc.get("metadata", {}).get("source", "未知")
        })
    
    # 显示表格（不可编辑） - 增加宽度
    st.markdown("""
        <style>
            section.main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    df = pd.DataFrame(results_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "序号": st.column_config.NumberColumn(width="small"),
            "内容预览": st.column_config.TextColumn(width="large"),
            "源文件": st.column_config.TextColumn(width="medium")
        }
    )
    
    # 使用选择框选择文档查看详情
    st.subheader("📄 文档详情")
    doc_options = [f"文档 {i+1} - {data['源文件']}" for i, data in enumerate(results_data)]
    selected_doc = st.selectbox(
        "选择文档查看详情",
        options=doc_options,
        key="doc_detail",
        index=doc_options.index(st.session_state.selected_doc) if st.session_state.selected_doc else 0,
        on_change=lambda: setattr(st.session_state, "selected_doc", st.session_state.doc_detail)
    )
    
    if selected_doc:
        doc_index = doc_options.index(selected_doc)
        selected_doc_data = docs[doc_index]
        st.session_state.selected_doc = selected_doc
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**元数据**")
            metadata = selected_doc_data.get("metadata", {})
            for key, value in metadata.items():
                if key not in ["vector", "embeddings"]:  # 过滤掉向量数据
                    st.write(f"**{key}**: {value}")
        
        with col2:
            st.write("**内容**")
            st.text_area(
                "文档内容",
                value=selected_doc_data.get("page_content", ""),
                height=300,
                key=f"doc_content_{doc_index}"
            )
